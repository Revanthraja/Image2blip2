# Image Captioning with BLIP2

This repository contains a Python script for processing images using the BLIP2 model to generate captions. The script reads images from a specified directory, generates captions using the BLIP2 model, and saves the processed data into two separate files—one for images and another for captions.

## Table of Contents

- [Features](#features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [License](#license)

## Features

- Process images and generate captions using the BLIP2 model.
- Save processed data into separate files for images and captions.
- Command-line arguments to customize the processing behavior.

## Getting Started

### Prerequisites

- Python (3.6 or later)
- Pip (Python package installer)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/YourUsername/image-captioning-blip2.git
   cd image-captioning-blip2
Install the required packages using the requirements.txt file:

## Usage
To process images and generate captions, run the script with appropriate command-line arguments. For example:
python process_images.py --image_directory path/to/your/images --clip_frame_data --beam_amount 5 --prompt_amount 10 --min_prompt_length 5 --max_prompt_length 20 --save_dir path/to/save/directory

Replace placeholders with actual paths and values. Refer to the Configuration section for details on available command-line arguments.

## Configuration
The script supports several command-line arguments for customization. Here are some of the available options:

--image_directory: Path to the directory containing images.
--clip_frame_data: Save images as clips to HDD/SSD.
--beam_amount: Amount for BLIP beam search.
--prompt_amount: Number of prompts per image.
--min_prompt_length: Minimum words required in prompt.
--max_prompt_length: Maximum words required in prompt.
--save_dir: Directory to save the processed data.
Refer to the script's help documentation for more information:

python process_images.py --help

## License
This project is licensed under the MIT License.
