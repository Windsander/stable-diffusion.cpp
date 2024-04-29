# Stable Diffusion Server

This project is a C++ implementation of a Stable Diffusion server that generates images based on textual prompts and optional input images. The server supports various generation modes, including text-to-image (txt2img), image-to-image (img2img), and image-to-video (img2vid).

## Features

- Generate images from textual prompts using the Stable Diffusion model
- Perform image-to-image generation by providing an initial image
- Create videos from a sequence of generated images
- Customize generation parameters such as image dimensions, number of inference steps, and guidance scale
- Support for different image formats (PNG, JPEG, etc.)
- API endpoint for easy integration with other applications


## Usage

1. Start the Stable Diffusion server:
   ```
   ./server --model /path/to/model --vae /path/to/vae --taesd /path/to/taesd --controlnet /path/to/controlnet --embd-dir /path/to/embeddings --lora-model-dir /path/to/lora_models
   ```

   The server will start running on `http://localhost:8080`.

2. Make API requests to generate images:

   - **Text-to-Image (txt2img)**:
     ```
     curl -X POST -H "Content-Type: application/json" -d '{
       "prompt": "a lovely cat",
       "width": 1024,
       "height": 1024,
       "num_inference_steps": 10,
       "guidance_scale": 7.5
     }' http://localhost:8080/generate --output test.png
     ```

   - **Image-to-Image (img2img)**:
     ```
     # Preprocess the input image and convert it to base64
     base64_image=$(base64 input_image.png)

     curl -X POST -H "Content-Type: application/json" -d '{
       "mode": "img2img",
       "input_image": "data:image/png;base64,'"$base64_image"'",
       "prompt": "a cat in a fantasy landscape",
       "width": 1024,
       "height": 1024,
       "num_inference_steps": 10,
       "guidance_scale": 7.5,
       "strength": 0.8
     }' http://localhost:8080/generate --output output_image.png
     ```

   The generated image will be saved as `test.png` (txt2img) or `output_image.png` (img2img) in the current directory.

## API Reference

### POST /generate

Generate an image based on the provided parameters.

#### Request Parameters

| Parameter           | Type   | Description                                                    |
|:-------------------|:-------|:---------------------------------------------------------------|
| mode               | string | Generation mode: "txt2img" (default), "img2img", or "img2vid"  |
| input_image        | string | Base64-encoded input image for img2img mode                    |
| prompt             | string | Textual prompt for image generation                            |
| negative_prompt    | string | Negative prompt to guide image generation (optional)           |
| width              | int    | Width of the generated image (default: 512)                    |
| height             | int    | Height of the generated image (default: 512)                   |
| num_inference_steps| int    | Number of inference steps for image generation (default: 20)   |
| guidance_scale     | float  | Scale for guided image generation (default: 7.5)               |
| strength           | float  | Strength of the img2img generation (default: 0.8)              |
| seed               | int    | Random seed for image generation (default: random)             |

#### Response

The generated image will be returned as the response body with the appropriate image format (e.g., PNG, JPEG).

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- [Stable Diffusion](https://github.com/CompVis/stable-diffusion) - Official Stable Diffusion repository
- [httplib](https://github.com/yhirose/cpp-httplib) - C++ HTTP server library
- [RapidJSON](https://github.com/Tencent/rapidjson) - JSON parser/generator for C++

Feel free to customize and expand upon this README file based on your specific project requirements and additional features.