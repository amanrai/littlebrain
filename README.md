# littlebrain

This repo trains ablations of very small transformer based auto-regressive models, to understand the impact of modelling choices on net usability of models. 
Ablations planned:
1. High depth (32 layers), low-width (128) transformers. 
2. High width (1024), low-depth (6 layers) transformers. 
3. High embedding dim (1024), low-network dim (128), high-depth (32 layers)
4. Low embedding dim (128), high-network dim (1024), high-depth (32 layers)
5. High embedding dim (1024), low-network dim(128), low-depth (6 layers)

- Min. Gradient updates per network: 5000
- Net tokens seen per network: bs * grad_accum_steps * 5000; Attempt will be to keep net tokens seen across networks stable. 

Pre-training Dataset: Dolma
Fine-tuning Dataset: Orca-2
