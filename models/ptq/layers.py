import torch
import torch.nn as nn
from torch.nn import functional as F

from typing import Optional, Tuple
from torch.nn.parameter import Parameter
from torch.nn.modules.linear import _LinearWithBias
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_

from .bit_type import BIT_TYPE_DICT
from .observer import build_observer
from .quantizer import build_quantizer

Tensor = torch.Tensor

'''class NonDynamicallyQuantizableLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)'''

class QConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=True,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform"):
        super(QConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = "conv_weight"
        self.observer = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            # print(self.calibrate)
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )
        weight = self.quantizer(self.weight)
        return F.conv2d(
            x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )


class QLinear(nn.Linear):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform"):
        super(QLinear, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = "linear_weight"
        self.observer = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer(self.weight)
        return F.linear(x, weight, self.bias)


class QAct(nn.Module):
    def __init__(
            self,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform"):
        super(QAct, self).__init__()

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = "activation"
        self.observer = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    def forward(self, x):
        if self.calibrate:
            self.quantizer.observer.update(x)
            if self.last_calibrate:
                # import ipdb;ipdb.set_trace()
                self.quantizer.update_quantization_params(x)
        if not self.quant:
            return x
        x = self.quantizer(x)
        return x


class QIntLayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(QIntLayerNorm, self).__init__(
            normalized_shape, eps, elementwise_affine)
        assert isinstance(normalized_shape, int)
        self.mode = 'ln'

    def get_MN(self, x):
        bit = 7
        N = torch.clamp(bit - torch.floor(torch.log2(x)), 0, 31)
        M = torch.clamp(torch.floor(x * torch.pow(2, N)), 0, 2**(bit+1)-1)
        return M, N

    def forward(self, x, in_quantizer=None, out_quantizer=None, in_scale_expand=1):
        if True: # # self.mode == 'ln':
            x = F.layer_norm(x, self.normalized_shape,
                             self.weight, self.bias, self.eps)
        elif self.mode == 'int':
            in_scale = in_quantizer.scale
            if in_scale_expand != 1:
                in_scale = in_scale.unsqueeze(-1).expand(-1,
                                                         in_scale_expand).T.reshape(-1)
            out_scale = out_quantizer.scale
            try:
                assert in_scale is not None and out_scale is not None
            except:
                import pdb;pdb.set_trace()
            channel_nums = x.shape[-1]
            in_scale = in_scale.reshape(1, 1, -1)
            out_scale = out_scale.reshape(1, 1, -1)
            x_q = (x / in_scale).round()
            in_scale1 = in_scale.min()
            in_scale_mask = (in_scale / in_scale1).round()

            x_q = x_q * in_scale_mask

            mean_x_q = x_q.mean(dim=-1) * in_scale1
            std_x_q = (in_scale1 / channel_nums) * torch.sqrt(channel_nums *
                                                              (x_q ** 2).sum(dim=-1) - x_q.sum(dim=-1) ** 2)

            A = (in_scale1 / std_x_q).unsqueeze(-1) * \
                self.weight.reshape(1, 1, -1) / out_scale
            A_sign = A.sign()
            M, N = self.get_MN(A.abs())
            B = ((self.bias.reshape(1, 1, -1) - (mean_x_q / std_x_q).unsqueeze(-1)
                 * self.weight.reshape(1, 1, -1)) / out_scale * torch.pow(2, N)).round()

            x_q = ((A_sign * M * x_q + B) / torch.pow(2, N)).round()
            x = x_q * out_scale
        else:
            raise NotImplementedError
        return x


class QIntSoftmax(nn.Module):
    def __init__(
            self,
            log_i_softmax=False,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform"):
        super(QIntSoftmax, self).__init__()

        self.log_i_softmax = log_i_softmax
        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = "activation"
        self.observer = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.quantizer = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer, self.module_type)

    @staticmethod
    def log_round(x):
        x_log_floor = x.log2().floor()
        big = x_log_floor
        extra_mask = (x - 2**big) >= 2**(big - 1)
        big[extra_mask] = big[extra_mask] + 1
        return big

    @staticmethod
    def int_softmax(x, scaling_factor, calibration_mode):
        def int_polynomial(x_int, scaling_factor):
            coef = [0.35815147, 0.96963238, 1.]  # ax**2 + bx + c
            coef[1] /= coef[0]
            coef[2] /= coef[0]
            b_int = torch.floor(coef[1] / scaling_factor)
            c_int = torch.floor(coef[2] / scaling_factor ** 2)
            z = x_int + b_int
            z = x_int * z
            z = z + c_int
            scaling_factor = coef[0] * scaling_factor ** 2
            return z, scaling_factor

        def int_exp(x_int, scaling_factor):
            x0 = -0.6931  # -ln2
            n = 30  # sufficiently large integer
            x0_int = torch.floor(x0 / scaling_factor)
            x_int = torch.max(x_int, n * x0_int)
            q = torch.floor(x_int / x0_int)
            r = x_int - x0_int * q
            exp_int, exp_scaling_factor = int_polynomial(r, scaling_factor)
            exp_int = torch.clamp(torch.floor(exp_int * 2 ** (n - q)), min=0)
            scaling_factor = exp_scaling_factor / 2 ** n
            return exp_int, scaling_factor

        # import pdb;pdb.set_trace()
        if calibration_mode == 'channel_wise':
            if len(x.shape)==4:
                scaling_factor = scaling_factor.unsqueeze(0).unsqueeze(2).unsqueeze(3)

        try:
            x_int = x/scaling_factor
        except:
            import pdb;pdb.set_trace()

        x_int_max, _ = x_int.max(dim=-1, keepdim=True)
        x_int = x_int - x_int_max
        exp_int, exp_scaling_factor = int_exp(x_int, scaling_factor)
        exp_int_sum = exp_int.sum(dim=-1, keepdim=True)
        return exp_int, exp_int_sum

    def forward(self, x, scale):
        if self.log_i_softmax and scale is not None:
            exp_int, exp_int_sum = self.int_softmax(x, scale, self.calibration_mode)
            softmax_out = torch.round(exp_int_sum / exp_int)
            rounds = self.log_round(softmax_out)
            mask = rounds >= 2**self.bit_type.bits
            qlog = torch.clamp(rounds, 0, 2**self.bit_type.bits-1)
            deq_softmax = 2**(-qlog)
            deq_softmax[mask] = 0
            return deq_softmax
        else:
            x = x.softmax(dim=-1)
            if self.calibrate:
                self.quantizer.observer.update(x)
                if self.last_calibrate:
                    self.quantizer.update_quantization_params(x)
            if not self.quant:
                return x
            x = self.quantizer(x)
            return x

class QLinear_Act(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type_w=BIT_TYPE_DICT["int8"],
            calibration_mode_w="layer_wise",
            observer_str_w="minmax",
            quantizer_str_w="uniform", 
            bit_type_a=BIT_TYPE_DICT["int8"],
            calibration_mode_a="layer_wise",
            observer_str_a="minmax",
            quantizer_str_a="uniform"):
        super(QLinear_Act, self).__init__(in_features, out_features, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type_w = bit_type_w
        self.bit_type_a = bit_type_a
        self.calibration_mode_w = calibration_mode_w
        self.calibration_mode_a = calibration_mode_a
        self.observer_str_w = observer_str_w
        self.observer_str_a = observer_str_a
        self.quantizer_str_w = quantizer_str_w
        self.quantizer_str_a = quantizer_str_a

        self.module_type_w = "linear_weight"
        self.observer_w = build_observer(
            self.observer_str_w, self.module_type_w, self.bit_type_w, self.calibration_mode_w)
        self.quantizer_w = build_quantizer(self.quantizer_str_w, self.bit_type_w,
                                         self.observer_w, self.module_type_w)  


        self.module_type_a = "activation"
        self.observer_a = build_observer(
            self.observer_str_a, self.module_type_a, self.bit_type_a, self.calibration_mode_a)
        self.quantizer_a = build_quantizer(self.quantizer_str_a, self.bit_type_a,
                                         self.observer_a, self.module_type_a)

    def forward(self, x):
        # Quantize Weight
        if self.calibrate:
            self.quantizer_w.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer_w.update_quantization_params(x)
        if not self.quant:
            return F.linear(x, self.weight, self.bias)
        weight = self.quantizer_w(self.weight)
        x = F.linear(x, weight, self.bias)

        # Quantize Activation
        if self.calibrate:
            self.quantizer_a.observer.update(x)
            if self.last_calibrate:
                # import ipdb;ipdb.set_trace()
                self.quantizer_a.update_quantization_params(x)
        if not self.quant:
            return x
        x = self.quantizer_a(x)
        return x
        
class QAttention(nn.Module):
    def __init__(
            self,
            cfg, 
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
            quant=False,
            calibrate=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = QLinear(
            dim,
            dim * 3,
            bias=qkv_bias,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_W,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W
        )
        self.qact1 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
        )
        self.qact2 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
        )
        self.proj = QLinear(
            dim,
            dim,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_W,
            calibration_mode=cfg.CALIBRATION_MODE_W,
            observer_str=cfg.OBSERVER_W,
            quantizer_str=cfg.QUANTIZER_W
        )
        self.qact3 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
        )
        self.qact_attn1 = QAct(
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_A,
            calibration_mode=cfg.CALIBRATION_MODE_A,
            observer_str=cfg.OBSERVER_A,
            quantizer_str=cfg.QUANTIZER_A
        )
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.log_int_softmax = QIntSoftmax(
            log_i_softmax=cfg.INT_SOFTMAX,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_S,
            calibration_mode=cfg.CALIBRATION_MODE_S,
            observer_str=cfg.OBSERVER_S,
            quantizer_str=cfg.QUANTIZER_S
        )

    def forward(self, x):
        B, N, C = x.shape
        x = self.qkv(x)
        x = self.qact1(x)
        qkv = x.reshape(B, N, 3, self.num_heads, C //
                        self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.qact_attn1(attn)
        attn = self.log_int_softmax(attn, self.qact_attn1.quantizer.scale)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.qact2(x)
        x = self.proj(x)
        x = self.qact3(x)
        x = self.proj_drop(x)
        return x


    def forward(self, x):
        if self.calibrate:
            # print(self.calibrate)
            self.quantizer.observer.update(self.weight)
            if self.last_calibrate:
                self.quantizer.update_quantization_params(x)

        weight = self.quantizer(self.weight)


'''class QMultiheadAttention(nn.MultiheadAttention):
    def __init__(
            self,
            embed_dim,
            num_heads,
            bias=True,
            quant=False,
            calibrate=False,
            last_calibrate=False,
            bit_type=BIT_TYPE_DICT["int8"],
            calibration_mode="layer_wise",
            observer_str="minmax",
            quantizer_str="uniform"):
        super(QMultiheadAttention, self).__init__(embed_dim, num_heads, bias)

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        self.module_type = "linear_weight"
        
        self.observer_in = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_out = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_q = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_k = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_v = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)

        self.quantizer_in = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_in, self.module_type)
        self.quantizer_out = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_out, self.module_type)
        self.quantizer_q = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_q, self.module_type)
        self.quantizer_k = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_k, self.module_type)
        self.quantizer_v = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_v, self.module_type)

    def forward(self, query, key, value, need_weights, attn_mask):
        
        if self.calibrate:
            self.quantizer_in.observer.update(self.in_proj.weight)
            self.quantizer_out.observer.update(self.out_proj.weight)
            self.quantizer_q.observer.update(self.q_proj_weight)
            self.quantizer_k.observer.update(self.k_proj_weight)
            self.quantizer_v.observer.update(self.v_proj_weight)

            if self.last_calibrate:
                self.quantizer_in.update_quantization_params(x)
                self.quantizer_out.update_quantization_params(x)
                self.quantizer_q.update_quantization_params(x)
                self.quantizer_k.update_quantization_params(x)
                self.quantizer_v.update_quantization_params(x)

        if not self.quant:
            return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)

        in_proj_weight = self.quantizer_in(self.in_proj_weight)
        out_proj.weight = self.quantizer_out(self.out_proj.weight)
        q_proj_weight = self.quantizer_q(self.q_proj_weight)
        k_proj_weight = self.quantizer_k(self.k_proj_weight)
        v_proj_weight = self.quantizer_v(self.v_proj_weight)

        return F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, out_proj.weight, self.out_proj.bias,
                training=self.training, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight)

'''
class QMultiheadAttention(nn.MultiheadAttention):
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, 
                quant=False,
                calibrate=False,
                last_calibrate=False,
                bit_type=BIT_TYPE_DICT["int8"],
                calibration_mode="layer_wise",
                observer_str="minmax",
                quantizer_str="uniform", 
                bit_type_a=BIT_TYPE_DICT["int8"],
                calibration_mode_a="channel_wise",
                observer_str_a="minmax",
                quantizer_str_a="uniform"):
        super(QMultiheadAttention, self).__init__(embed_dim=embed_dim, num_heads=num_heads)

        '''self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = _LinearWithBias(embed_dim, embed_dim)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()'''

        self.quant = quant
        self.calibrate = calibrate
        self.last_calibrate = last_calibrate
        self.bit_type = bit_type
        self.calibration_mode = calibration_mode
        self.observer_str = observer_str
        self.quantizer_str = quantizer_str

        '''# if self.quant:
        self.c_act = QAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=bit_type_a,
            calibration_mode=calibration_mode_a,
            observer_str=observer_str_a,
            quantizer_str=quantizer_str_a
        )
        self.q_act = QAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=bit_type_a,
            calibration_mode=calibration_mode_a,
            observer_str=observer_str_a,
            quantizer_str=quantizer_str_a
        )
        self.q_act_1 = QAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=bit_type_a,
            calibration_mode=calibration_mode_a,
            observer_str=observer_str_a,
            quantizer_str=quantizer_str_a
        )
        self.k_act_1 = QAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=bit_type_a,
            calibration_mode=calibration_mode_a,
            observer_str=observer_str_a,
            quantizer_str=quantizer_str_a
        )
        self.v_act_1 = QAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=bit_type_a,
            calibration_mode=calibration_mode_a,
            observer_str=observer_str_a,
            quantizer_str=quantizer_str_a
        )
        self.attn_act = QAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=bit_type_a,
            calibration_mode=calibration_mode_a,
            observer_str=observer_str_a,
            quantizer_str=quantizer_str_a
        )
        self.k_act = QAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=bit_type_a,
            calibration_mode=calibration_mode_a,
            observer_str=observer_str_a,
            quantizer_str=quantizer_str_a
        )
        self.v_act = QAct(
            quant=self.quant,
            calibrate=self.calibrate,
            bit_type=bit_type_a,
            calibration_mode=calibration_mode_a,
            observer_str=observer_str_a,
            quantizer_str=quantizer_str_a
        )'''

        self.module_type = "linear_weight"
        
        self.observer_in = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_out = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_q = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_k = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)
        self.observer_v = build_observer(
            self.observer_str, self.module_type, self.bit_type, self.calibration_mode)

        self.quantizer_in = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_in, self.module_type)
        self.quantizer_out = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_out, self.module_type)
        self.quantizer_q = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_q, self.module_type)
        self.quantizer_k = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_k, self.module_type)
        self.quantizer_v = build_quantizer(self.quantizer_str, self.bit_type,
                                         self.observer_v, self.module_type)

    '''def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(QMultiheadAttention, self).__setstate__(state)'''
        
    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):

        if self.calibrate:
            if self._qkv_same_embed_dim:
                self.quantizer_in.observer.update(self.in_proj_weight)
                # self.quantizer_out.observer.update(self.out_proj.weight)
                if self.last_calibrate:
                    self.quantizer_in.update_quantization_params(query)
            else:
                self.quantizer_q.observer.update(self.q_proj_weight)
                self.quantizer_k.observer.update(self.k_proj_weight)
                self.quantizer_v.observer.update(self.v_proj_weight)

                if self.last_calibrate:
                    self.quantizer_in.update_quantization_params(query)
                    # self.quantizer_out.update_quantization_params(query)
                    self.quantizer_q.update_quantization_params(query)
                    self.quantizer_k.update_quantization_params(key)
                    self.quantizer_v.update_quantization_params(value)
        
        if not self._qkv_same_embed_dim:
            if not self.quant:
                return F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, use_separate_proj_weight=True,
                    q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                    v_proj_weight=self.v_proj_weight)
            else:
                in_proj_weight = self.quantizer_in(self.in_proj_weight)
                out_proj_weight = self.quantizer_out(self.out_proj.weight)
                q_proj_weight = self.quantizer_q(self.q_proj_weight)
                k_proj_weight = self.quantizer_k(self.k_proj_weight)
                v_proj_weight = self.quantizer_v(self.v_proj_weight)
                return F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, out_proj_weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, use_separate_proj_weight=True,
                    q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                    v_proj_weight=v_proj_weight)
                '''quant_multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, use_separate_proj_weight=True,
                    q_act=self.q_act, q_act_1=self.q_act_1, 
                    k_act_1=self.k_act_1, v_act_1=self.v_act_1, 
                    attn_act=self.attn_act, 
                    q_proj_weight=q_proj_weight, k_act=self.k_act, 
                    k_proj_weight=k_proj_weight, v_act=self.v_act, 
                    v_proj_weight=v_proj_weight)'''

        else:
            if not self.quant:
                return F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    self.in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, self.out_proj.weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask)
            else:
                in_proj_weight = self.quantizer_in(self.in_proj_weight)
                out_proj_weight = self.quantizer_in(self.out_proj.weight)
                # q_proj_weight = self.quantizer_q(self.q_proj_weight)
                # k_proj_weight = self.quantizer_k(self.k_proj_weight)
                # v_proj_weight = self.quantizer_v(self.v_proj_weight)
                return F.multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, out_proj_weight, self.out_proj.bias,
                    training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask)
                '''quant_multi_head_attention_forward(
                    query, key, value, self.embed_dim, self.num_heads,
                    in_proj_weight, self.in_proj_bias,
                    self.bias_k, self.bias_v, self.add_zero_attn,
                    self.dropout, out_proj_weight, self.out_proj.bias,
                    c_act=self.c_act, training=self.training,
                    key_padding_mask=key_padding_mask, need_weights=need_weights,
                    attn_mask=attn_mask, q_act=self.q_act, q_act_1=self.q_act_1, 
                    k_act_1=self.k_act_1, v_act_1=self.v_act_1, 
                    attn_act=self.attn_act, k_act=self.k_act, v_act=self.v_act)'''
        
Tensor = torch.Tensor
def quant_multi_head_attention_forward(query: Tensor,
                                 key: Tensor,
                                 value: Tensor,
                                 embed_dim_to_check: int,
                                 num_heads: int,
                                 in_proj_weight: Tensor,
                                 in_proj_bias: Tensor,
                                 bias_k: Optional[Tensor],
                                 bias_v: Optional[Tensor],
                                 add_zero_attn: bool,
                                 dropout_p: float,
                                 out_proj_weight: Tensor,
                                 out_proj_bias: Tensor,
                                 c_act: Optional[nn.Module] = None,
                                 training: bool = True,
                                 key_padding_mask: Optional[Tensor] = None,
                                 need_weights: bool = True,
                                 attn_mask: Optional[Tensor] = None,
                                 use_separate_proj_weight: bool = False,
                                 q_act: Optional[nn.Module] = None,
                                 q_act_1: Optional[nn.Module] = None,
                                 k_act_1: Optional[nn.Module] = None,
                                 v_act_1: Optional[nn.Module] = None,
                                 attn_act: Optional[nn.Module] = None,
                                 q_proj_weight: Optional[Tensor] = None,
                                 k_act: Optional[nn.Module] = None,
                                 k_proj_weight: Optional[Tensor] = None,
                                 v_act: Optional[nn.Module] = None,
                                 v_proj_weight: Optional[Tensor] = None,
                                 static_k: Optional[Tensor] = None,
                                 static_v: Optional[Tensor] = None
                                 ) -> Tuple[Tensor, Optional[Tensor]]:
    
    if not torch.jit.is_scripting():
        tens_ops = (query, key, value, in_proj_weight, in_proj_bias, bias_k, bias_v,
                    out_proj_weight, out_proj_bias)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                multi_head_attention_forward, tens_ops, query, key, value,
                embed_dim_to_check, num_heads, in_proj_weight, in_proj_bias,
                bias_k, bias_v, add_zero_attn, dropout_p, out_proj_weight,
                out_proj_bias, training=training, key_padding_mask=key_padding_mask,
                need_weights=need_weights, attn_mask=attn_mask,
                use_separate_proj_weight=use_separate_proj_weight,
                q_proj_weight=q_proj_weight, k_proj_weight=k_proj_weight,
                v_proj_weight=v_proj_weight, static_k=static_k, static_v=static_v)
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    # allow MHA to have different sizes for the feature dimension
    assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            # self-attention
            q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif torch.equal(key, value):
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        query = q_act(query)
        key = k_act(key)
        value = v_act(value)

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if attn_mask is not None:
        assert attn_mask.dtype == torch.float32 or attn_mask.dtype == torch.float64 or \
            attn_mask.dtype == torch.float16 or attn_mask.dtype == torch.uint8 or attn_mask.dtype == torch.bool, \
            'Only float, byte, and bool types are supported for attn_mask, not {}'.format(attn_mask.dtype)
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)

        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    # convert ByteTensor key_padding_mask to bool
    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:
        warnings.warn("Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = pad(key_padding_mask, (0, 1))
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = pad(attn_mask, (0, 1))
        if key_padding_mask is not None:
            key_padding_mask = pad(key_padding_mask, (0, 1))
    
    q = q_act_1(q)
    k = k_act_1(k)
    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_output_weights.masked_fill_(attn_mask, float('-inf'))
        else:
            attn_output_weights += attn_mask


    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = dropout(attn_output_weights, p=dropout_p, training=training)
    
    attn_output_weights = attn_act(attn_output_weights)
    v = v_act_1(v)
    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    
    attn_output = c_act(attn_output)
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None
    