import torch
import torch.nn.functional as F
import lightning as L

from main import instantiate_from_config
from contextlib import contextmanager

from taming.modules.diffusionmodules.improved_model import Encoder, Decoder
from taming.modules.scheduler.lr_scheduler import Scheduler_LinearWarmup, Scheduler_LinearWarmup_CosineDecay
from taming.modules.util import requires_grad
from collections import OrderedDict
from taming.modules.ema import LitEma

import numpy as np

class VQModel(L.LightningModule):
    def __init__(self,
                 ddconfig,
                 lossconfig,
                 ### Quantize Related
                 quantconfig,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 learning_rate=None,
                 ### scheduler config
                 warmup_epochs=1.0, #warmup epochs
                 scheduler_type = "linear-warmup_cosine-decay",
                 accumulate_steps = 1,
                 min_learning_rate = 0,
                 use_ema = False,
                 stage = None,
                 ):
        super().__init__()
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = instantiate_from_config(quantconfig)
        self.use_ema = use_ema
        self.stage = stage
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, stage=stage)
        self.image_key = image_key
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        if monitor is not None:
            self.monitor = monitor

        if self.use_ema and stage is None: #no need to construct ema when training transformer
            self.model_ema = LitEma(self)
        self.learning_rate = learning_rate
        self.scheduler_type = scheduler_type
        self.warmup_epochs = warmup_epochs
        self.min_learning_rate = min_learning_rate
        self.automatic_optimization = False
        self.accumulate_steps = accumulate_steps

        self.strict_loading = False
        self.example_input_array = (torch.randn(1, 3, 128, 128))
        self.codebook_count = [0] * self.quantize.n_e

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        '''
        save the state_dict and filter out the 
        '''
        return {k: v for k, v in super().state_dict(*args, destination, prefix, keep_vars).items() if ("inception_model" not in k and "lpips_vgg" not in k and "lpips_alex" not in k)}
        
    def init_from_ckpt(self, path, ignore_keys=list(), stage=None):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        ema_mapping = {}
        new_params = OrderedDict()
        if stage == "transformer": ### directly use ema encoder and decoder parameter
            if self.use_ema:
                for k, v in sd.items(): 
                    if "encoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v   
                        s_name = k.replace('.', '')
                        ema_mapping.update({s_name: k})
                        continue
                    if "decoder" in k:
                        if "model_ema" in k:
                            k = k.replace("model_ema.", "") #load EMA Encoder or Decoder
                            new_k = ema_mapping[k]
                            new_params[new_k] = v 
                        s_name = k.replace(".", "")
                        ema_mapping.update({s_name: k})
                        continue 
            else: #also only load the Generator
                for k, v in sd.items():
                    if "encoder" in k:
                        new_params[k] = v
                    elif "decoder" in k:
                        new_params[k] = v
            missing_keys, unexpected_keys = self.load_state_dict(new_params, strict=False)
        else: ## simple resume
            missing_keys, unexpected_keys = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        (quant, emb_loss, info), loss_breakdown = self.quantize(h)
        ### using token factorization the info is a tuple (each for embedding)
        return quant, emb_loss, info, loss_breakdown

    def decode(self, quant):
        dec = self.decoder(quant)
        return dec

    def forward(self, input):
        quant, diff, indices, loss_break = self.encode(input)
        dec = self.decode(quant)
        for ind in indices.unique():
            self.codebook_count[ind] = 1
        return dec, diff, loss_break

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format)
        return x.float()

    # fix mulitple optimizer bug
    # refer to https://lightning.ai/docs/pytorch/stable/model/manual_optimization.html
    def training_step(self, batch, batch_idx):
        x = self.get_input(batch, self.image_key)
        xrec, eloss, loss_break = self(x)

        opt_gen, opt_disc = self.optimizers()
        if self.scheduler_type != "None":
            scheduler_gen, scheduler_disc = self.lr_schedulers()

        ####################
        # fix global step bug
        # refer to https://github.com/Lightning-AI/pytorch-lightning/issues/17958
        opt_disc._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        opt_disc._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        # opt_gen._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
        # opt_gen._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")
        ####################
        
        # optimize generator
        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, xrec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        aeloss = aeloss / self.accumulate_steps
        self.manual_backward(aeloss)
        
        if (batch_idx + 1) % self.accumulate_steps == 0:
            opt_gen.step()
            opt_gen.zero_grad()
            if self.scheduler_type != "None":
                scheduler_gen.step()
        
        log_dict_ae["train/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
            
        # optimize discriminator
        discloss, log_dict_disc = self.loss(eloss, loss_break, x, xrec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        discloss = discloss / self.accumulate_steps
        self.manual_backward(discloss)
        
        if (batch_idx + 1) % self.accumulate_steps == 0:
            opt_disc.step()
            opt_disc.zero_grad()
            if self.scheduler_type != "None":
                scheduler_disc.step()
            
        #if torch.distributed.get_rank() == 0:
        #    print(log_dict_ae, log_dict_disc)

        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)
            
    def on_train_epoch_start(self):
        self.codebook_count = [0] * self.quantize.n_e
        
    def on_validation_epoch_start(self):
        self.codebook_count = [0] * self.quantize.n_e

    def validation_step(self, batch, batch_idx): 
        if self.use_ema:
            with self.ema_scope():
                log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
        else:
            log_dict = self._validation_step(batch, batch_idx)

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        quant, eloss, indices, loss_break = self.encode(x)
        x_rec = self.decode(quant).clamp(-1, 1)
        aeloss, log_dict_ae = self.loss(eloss, loss_break, x, x_rec, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="val"+ suffix)

        discloss, log_dict_disc = self.loss(eloss, loss_break, x, x_rec, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="val" + suffix)
        
        for ind in indices.unique():
            self.codebook_count[ind] = 1
        log_dict_ae[f"val{suffix}/codebook_util"] = torch.tensor(sum(self.codebook_count) / len(self.codebook_count))
    
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        if batch_idx % 100 == 0:
            # x is of shape (B, C, H, W)
            stacked = torch.cat([x, x_rec], dim=-2)
            stacked = (stacked * 127.5 + 127.5).to(torch.uint8)
            self.logger.log_image(key=f"val/batch={batch_idx}", images=[stacked[0]])

        return self.log_dict

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_gen = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        if self.trainer.is_global_zero:
            print("step_per_epoch: {}".format(len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size))
        step_per_epoch  = len(self.trainer.datamodule._train_dataloader()) // self.trainer.world_size
        warmup_steps = step_per_epoch * self.warmup_epochs
        training_steps = step_per_epoch * self.trainer.max_epochs

        if self.scheduler_type == "None":
            return ({"optimizer": opt_gen}, {"optimizer": opt_disc})
    
        if self.scheduler_type == "linear-warmup":
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup(warmup_steps))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup(warmup_steps))

        elif self.scheduler_type == "linear-warmup_cosine-decay":
            multipler_min = self.min_learning_rate / self.learning_rate
            scheduler_ae = torch.optim.lr_scheduler.LambdaLR(opt_gen, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
            scheduler_disc = torch.optim.lr_scheduler.LambdaLR(opt_disc, Scheduler_LinearWarmup_CosineDecay(warmup_steps=warmup_steps, max_steps=training_steps, multipler_min=multipler_min))
        else:
            raise NotImplementedError()
        return {"optimizer": opt_gen, "lr_scheduler": scheduler_ae}, {"optimizer": opt_disc, "lr_scheduler": scheduler_disc}

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x                                       
