# 2D-Human-Parsing

![2D-Huamn-Parsing](/assets/teaser.png "Teaser Image")

## Requirements

```
python == 3.8.8, pytorch == 1.7.1, torchvision == 0.8.2, networkx == 2.6.2
```

## Demo Usage
Download the pretrained model [here](https://drive.google.com/drive/folders/1ZvXgp8EdcoHFu9uici7jDtin6hi_VO3h?usp=sharing) and put it to the `pretrained` folder, then simply run:
```sh
cd inference
bash demo.sh
```

## Testing on custom data

To test on your own data, just refer to the structure of the `demo_imgs` folder and the `inference/demo.sh` file respectively for data preparation and model running. Guess it would be very easy to get start by replacing them with your own data / bash script :).

## Parsing Index
Below is the person part index (i.e. pixel value) of the parsing result:
|  Part   | index | Part | index |
|  ----  | ----  |  ----  | ----  |
| background  | 0 | neck | 10 |
| hat  | 1 | scarf | 11 |
| hair  | 2 | skirt | 12 |
| - | 3 | face | 13 |
| sunglass  | 4 | left arm | 14 |
| shirt  | 5 | right arm | 15 |
| dress  | 6 | left leg | 16 |
| coats  | 7 | right leg | 17 | 
| -  | 8 | left shoe | 18 |
| pant  | 9 | right shoe | 19 |

## Acknowledgement
The code and pretrained model in this repo are provided through the courtesy of [Bowen Wu](https://github.com/Bowenwu1). Thanks for his effort at making this easy-to-use human parsing codebase.

## Citation
If you find this repo helpful and use it in your work/paper, please cite:
```
@misc{BowenParsing2021,
  author = {Bowen Wu, Fuwei Zhao},
  title = {2D-Human-Parsing},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/fyviezhao/2D-Human-Parsing}}
}
```
