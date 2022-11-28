## from https://yandex.ru/clck/jsredir?from=yandex.ru%3Bsearch%2F%3Bweb%3B%3B&text=&etext=2202.VBk_rTLsC2TFALCwptHENLHmApeTWumS7-JljIljdC9wSLzRaLETmYseuywYIeflaG1sY3Npb3ptbGxmZ3huYg.e19f28dfd4d8b0c82e0bb9faadf40d222f9042e4&uuid=&state=jLT9ScZ_wbo,&&cst=AiuY0DBWFJ5fN_r-AEszkwhhXRGt787iQeVsX5XduQ5UGllTICnn-IZUz0sIfE9vqw_ROYXSANAaNW9nAY82QgkUWyAXRq3U1GLJE2ayhcsYsYEPyjysw6WunQXknhKSDjPYkHfkZJlIT5ne3QIqBZvq02NJG-2eDwf7NItDoeYCc-sl7t9AhAgCuOQdvcJWtF5LbXzr2eNqtivoKDO-f00Qyw7fvJmoy2lk7NtJjY-Nq7iiJnZx-vZCETR7vRweLdirO51eGQmkdFXUPTqpgaPPW3A-TwNLIapQmYVKHfyFDQn91DFlzjEc9Qk1aR_JczYMKwywtshCGKReLyDgf6An2oWo49jmZWR1hrzVkCDrLAdHlvNyRifmtATPxqCQsrr1EsbYFiATLOil55wBX2QJrlGDTR2QWEe8ziPaZ5jRGxf6d2LX4OAtJDAudsNv92nFZfEZurEdyc9r36IHSlYB5hUFMBJJl4iqpseoPeTiiC8FSzb8NKfUtN1p26zODDPEjY69IExwUAr2tFxXiQ2-0nHCKkks4QyQmHAJ11_RXlxSl-rfgcZiuGwzhYtArPjeV0c2_begaBGY89wejj315SX5h3F4G2Yn-YhqtR2hMfZw__Cd58Y5b1CR_QctqX_ubtiD3eeOVdXDyjmjH37b2dKWvmzjqYMuma7joASCIP_AI8C-5xECCi6cAdpoRXdfTeD8_2tClGLALua9TpY6dYqH949zhzHEf6w4Tth40SAzI1tO12FdxhZ4YK2ncMpsUgBHBtlvd-1CKlWk9hWnVCpXCFIDHFiFyapwRL-9Q2PKZNkzJDbVtkBVYVnmNV0FWq3oGWpE7MT8ig626kjuLGwytUs638lQpDRxGMTCqHoC5y7q7x8QJ0dqsuzd5UgEI-Qxm2COpVnRJ21Q_jGZftKAVa4QOxCH9V2MxUjByMeibITl4NbIZ2_7nFmUrbUx4jGo10uV-OmM-CHRP9EvriQakn9LoEyeeDrrAIpV0IFOR27tkJcKaC9n1ksUFznXPvqnu_0zKUvrHMabvo-AXgCgsQNlqQoLhX8BMTTXG-bVmTeRb5pZWIP_2jlYPO_-WQnuwVgFwNFuAnxPsOAo5ooNYDvt3CZVu26yTPUpJd_LO75se6x7IsRP6Bo_bA8iiOuUrZwx6oQtBrZasOvY2f8GNj0zlWwQamrOUVyb-cR7ve3icg,,&data=UlNrNmk5WktYejY4cHFySjRXSWhXSlhmejI2U1RSQmZ5cHlEU2tiUjR1Rkl3bTNjanc4b3FQQXExeTd0RWtBSF9XdnkwaFVCUWVRTE9xRWhSc2U5c0J5enl0WmNvTkVaRGdweTR4TDNVRGYzcHVoQVJ2MmQtQVFRdzhRNDVqRWFQQjU5djVtM1BFMCw,&sign=907e6be316653f8319e48b7e15a80f68&keyno=0&b64e=2&ref=orjY4mGPRjk5boDnW0uvlrrd71vZw9kplPKtknI-wJ0cVtKVkroyyqjxGK1gt5upzLzeM-EkN_kzdtuXqnsypzsa-JgqcahS&l10n=ru&cts=1669666744225%40%40events%3D%5B%7B%22event%22%3A%22click%22%2C%22id%22%3A%222_m1ymw04-01%22%2C%22cts%22%3A1669666744225%2C%22fast%22%3A%7B%22organic%22%3A1%7D%2C%22service%22%3A%22web%22%2C%22event-id%22%3A%22lb18gx41e4%22%7D%5D&mc=4&hdtime=19711
import torch
import torch.nn as nn
import numpy as np


class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        output = torch.bmm(attn, v)

        return output, attn
