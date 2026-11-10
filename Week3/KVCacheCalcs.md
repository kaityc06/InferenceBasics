LLama 3 70B Architecture:
Number of layers: 80
Number of KV heads: 8 (uses grouped-query attention, so 64 query heads share 8 key/value heads)
Head dimension: 128 (8192 hidden size / 64 attention heads)
Bytes per number: 2

Number of bytes used by KV cache per token = 2 * 80 * 8 * 128 * 2 = 327680 bytes

For 8k tokens: 327680 * 8K = 2.6GB

Memory for weights: 70B * 2 = 140GB

Total memory: 142.6GB