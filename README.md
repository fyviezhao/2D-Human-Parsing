# 2D-Human-Parsing

## Requirements

```
python == 3.8.8, pytorch == 1.7.1, torchvision == 0.8.2, networkx == 2.6.2
```

## Demo Usage
Download the pretrained model [here](https://figshare.com/s/04de7175dd937cf638e3) and put it to the `pretrained` folder, then simply run:
```sh
cd inference
bash demo.sh
```

## Testing on custom data

To test on your own data, just refer to the structure of the `deom_imgs` folder and the `inference/demo.sh` file respectively for data preparation and model running. Guess it would be very easy to get start by replacing them with your own data / bash script :).

## Acknowledgement
The code and pretrained model in this repo are provided through the courtesy of [Bowen Wu](https://github.com/Bowenwu1). Thanks for his effort at making this easy-to-use human parsing codebase.