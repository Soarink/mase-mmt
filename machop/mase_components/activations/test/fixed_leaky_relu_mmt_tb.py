import torch

from torch import nn
from torch.autograd.function import InplaceFunction

import cocotb

from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

import os
import random
from pathlib import Path

import pytest

import cocotb
from cocotb.clock import Clock
from cocotb.runner import get_runner
from cocotb.triggers import FallingEdge

pytestmark = pytest.mark.simulator_required


# snippets
class MyClamp(InplaceFunction):
    @staticmethod
    def forward(ctx, input, min, max):
        return input.clamp(min=min, max=max)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


class MyRound(InplaceFunction):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        return input.round()

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input


# wrap through module to make it a function
my_clamp = MyClamp.apply
my_round = MyRound.apply


# fixed-point quantization with a bias
def quantize(x, bits, bias):  # bits = 32
    """Do linear quantization to input according to a scale and number of bits"""
    thresh = 2 ** (bits - 1)
    scale = 2**bias
    return my_clamp(my_round(x.mul(scale)), -thresh, thresh - 1).div(scale)


class VerificationCase:
    bitwidth = 32
    bias = 4
    num = 16
    bias = 4
    
    def __init__(self, samples=2):
        self.m = nn.LeakyReLU(negative_slope=0.5**3)
        self.inputs, self.outputs = [], []
        for _ in range(samples):
            i, o = self.single_run()
            self.inputs.append(i)
            self.outputs.append(o)
        self.samples = samples


    def single_run(self):
        xs = torch.rand(self.num)
        r1, r2 = 4, -4
        xs = (r1 - r2) * xs + r2
        # 8-bit, (5, 3)
        xs = quantize(xs, self.bitwidth, self.bias)
        return xs, self.m(xs)

    def get_dut_parameters(self):
        return {
            "IN_SIZE": self.num,
            "IN_WIDTH": self.bitwidth,
            # "BIAS": self.bias
        }

    def get_dut_input(self, i):
        inputs = self.inputs[i]
        shifted_integers = (inputs * (2**self.bias)).int()
        return shifted_integers.numpy().tolist()

    def get_dut_output(self, i):
        outputs = self.outputs[i]
        # print('outputs= ', outputs.numpy().tolist())
        shifted_integers = (outputs * (2**self.bias))
        # print(shifted_integers)
        for j in range(len(shifted_integers)):
            tmp = shifted_integers[j]
            if (tmp < 0) and (int(tmp) != tmp):
                shifted_integers[j] -= 1
        # print(shifted_integers)
        shifted_integers = shifted_integers.int()
        return shifted_integers.numpy().tolist()


@cocotb.test()
async def test_fixed_leaky_relu_mmt(dut):
    """Test integer based Relu"""
    test_case = VerificationCase(samples=100)

    # set inputs outputs
    for i in range(test_case.samples):
        x = test_case.get_dut_input(i)
        y = test_case.get_dut_output(i)
        # print('x= ', x)
        # print('y= ', y)
        dut.data_in.value = x
        await Timer(2, units="ns")
        # print(x)
        # for i in range(len(y)):
        #     print('HW=', dut.data_out.value[i], ',\tSW=', y[i], ',\tResult=', dut.data_out.value[i]==signed_to_unsigned([y[i]])[0])
        assert dut.data_out.value == signed_to_unsigned(y), f"output q was incorrect on the {i}th cycle"


def signed_to_unsigned(x):
    ''' bitwidth=32 by default'''
    x_unsigned = []
    for e in x:    
        if e >= 0:
            x_unsigned.append(e)
        else:
            x_unsigned.append(2**32 - abs(e))
    return x_unsigned
    
def runner():
    sim = os.getenv("SIM", "verilator")
    verilog_sources = []

    verilog_sources = ["../../../../components/activations/fixed_leaky_relu_mmt.sv"]
    test_case = VerificationCase()

    # set parameters
    extra_args = []
    for k, v in test_case.get_dut_parameters().items():
        extra_args.append(f"-G{k}={v}")
    print(extra_args)
    runner = get_runner(sim)
    runner.build(
        verilog_sources=verilog_sources,
        hdl_toplevel="fixed_leaky_relu_mmt",
        build_args=extra_args,
    )

    runner.test(hdl_toplevel="fixed_leaky_relu_mmt", test_module="fixed_leaky_relu_mmt_tb")


if __name__ == "__main__":
    runner()
