#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <httplib.h>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

#include "stable-diffusion.h"  // Include the Stable Diffusion library header
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"

const char* rng_type_to_str[] = {
    "std_default",
    "cuda",
};

// Names of the sampler method, same order as enum sample_method in stable-diffusion.h
const char* sample_method_str[] = {
    "euler_a",
    "euler",
    "heun",
    "dpm2",
    "dpm++2s_a",
    "dpm++2m",
    "dpm++2mv2",
    "lcm",
};

// Names of the sigma schedule overrides, same order as sample_schedule in stable-diffusion.h
const char* schedule_str[] = {
    "default",
    "discrete",
    "karras",
};

const char* modes_str[] = {
    "txt2img",
    "img2img",
    "img2vid",
    "convert",
};

enum SDMode {
    TXT2IMG,
    IMG2IMG,
    IMG2VID,
    CONVERT,
    MODE_COUNT
};

struct SDParams {
    int n_threads = -1;
    SDMode mode   = TXT2IMG;

    std::string model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string esrgan_path;
    std::string controlnet_path;
    std::string embeddings_path;
    std::string stacked_id_embeddings_path;
    std::string input_id_images_path;
    sd_type_t wtype = SD_TYPE_COUNT;
    std::string lora_model_dir;
    std::string output_path = "output.png";
    std::string input_path;
    std::string control_image_path;

    std::string prompt;
    std::string negative_prompt;
    float min_cfg     = 1.0f;
    float cfg_scale   = 7.0f;
    float style_ratio = 20.f;
    int clip_skip     = -1;  // <= 0 represents unspecified
    int width         = 512;
    int height        = 512;
    int batch_count   = 1;

    int video_frames         = 6;
    int motion_bucket_id     = 127;
    int fps                  = 6;
    float augmentation_level = 0.f;

    sample_method_t sample_method = EULER_A;
    schedule_t schedule           = DEFAULT;
    int sample_steps              = 20;
    float strength                = 0.75f;
    float control_strength        = 0.9f;
    rng_type_t rng_type           = CUDA_RNG;
    int64_t seed                  = 42;
    bool verbose                  = false;
    bool vae_tiling               = false;
    bool control_net_cpu          = false;
    bool normalize_input          = false;
    bool clip_on_cpu              = false;
    bool vae_on_cpu               = false;
    bool canny_preprocess         = false;
    bool color                    = false;
    int upscale_repeats           = 1;
};

void print_params(SDParams params) {
    printf("Option: \n");
    printf("    n_threads:         %d\n", params.n_threads);
    printf("    mode:              %s\n", modes_str[params.mode]);
    printf("    model_path:        %s\n", params.model_path.c_str());
    printf("    wtype:             %s\n", params.wtype < SD_TYPE_COUNT ? sd_type_name(params.wtype) : "unspecified");
    printf("    vae_path:          %s\n", params.vae_path.c_str());
    printf("    taesd_path:        %s\n", params.taesd_path.c_str());
    printf("    esrgan_path:       %s\n", params.esrgan_path.c_str());
    printf("    controlnet_path:   %s\n", params.controlnet_path.c_str());
    printf("    embeddings_path:   %s\n", params.embeddings_path.c_str());
    printf("    stacked_id_embeddings_path:   %s\n", params.stacked_id_embeddings_path.c_str());
    printf("    input_id_images_path:   %s\n", params.input_id_images_path.c_str());
    printf("    style ratio:       %.2f\n", params.style_ratio);
    printf("    normzalize input image :  %s\n", params.normalize_input ? "true" : "false");
    printf("    output_path:       %s\n", params.output_path.c_str());
    printf("    init_img:          %s\n", params.input_path.c_str());
    printf("    control_image:     %s\n", params.control_image_path.c_str());
    printf("    clip on cpu:       %s\n", params.clip_on_cpu ? "true" : "false");
    printf("    controlnet cpu:    %s\n", params.control_net_cpu ? "true" : "false");
    printf("    vae decoder on cpu:%s\n", params.vae_on_cpu ? "true" : "false");
    printf("    strength(control): %.2f\n", params.control_strength);
    printf("    prompt:            %s\n", params.prompt.c_str());
    printf("    negative_prompt:   %s\n", params.negative_prompt.c_str());
    printf("    min_cfg:           %.2f\n", params.min_cfg);
    printf("    cfg_scale:         %.2f\n", params.cfg_scale);
    printf("    clip_skip:         %d\n", params.clip_skip);
    printf("    width:             %d\n", params.width);
    printf("    height:            %d\n", params.height);
    printf("    sample_method:     %s\n", sample_method_str[params.sample_method]);
    printf("    schedule:          %s\n", schedule_str[params.schedule]);
    printf("    sample_steps:      %d\n", params.sample_steps);
    printf("    strength(img2img): %.2f\n", params.strength);
    printf("    rng:               %s\n", rng_type_to_str[params.rng_type]);
    printf("    seed:              %ld\n", params.seed);
    printf("    video-total-frames: %d\n", params.video_frames);
    printf("    video-fps:          %d\n", params.fps);
    printf("    motion-bucket-id:   %d\n", params.motion_bucket_id);
    printf("    batch_count:       %d\n", params.batch_count);
    printf("    vae_tiling:        %s\n", params.vae_tiling ? "true" : "false");
    printf("    upscale_repeats:   %d\n", params.upscale_repeats);
}

class StableDiffusionServer {
public:
    StableDiffusionServer(const std::string& modelPath, const std::string& vaePath, int numThreads)
        : m_modelPath(modelPath), m_vaePath(vaePath), m_numThreads(numThreads) {}

    void Start(int port) {
        InitializeModel();
        SetupRoutes();
        m_server.listen("0.0.0.0", port);
    }

private:
    void InitializeModel() {
        m_sdContext = new_sd_ctx(m_modelPath.c_str(), m_vaePath.c_str(), nullptr, nullptr,
                                 nullptr, nullptr, nullptr, true, false, true, m_numThreads, SD_TYPE_F32,
                                 CUDA_RNG, DEFAULT, false, false, false);
        if (!m_sdContext) {
            throw std::runtime_error("Failed to initialize Stable Diffusion model");
        }
    }

    void SetupRoutes() {
        m_server.Post("/generate", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                SDParams params = ParseRequestData(req.body);
                sd_image_t* result = GenerateImage(params);
                if (result) {
                    res.set_content((const char*)result->data, result->width * result->height * result->channel, "image/png");
                    free(result->data);
                    free(result);
                } else {
                    res.status = 500;
                    res.set_content("Failed to generate image", "text/plain");
                }
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(e.what(), "text/plain");
            }
        });
    }

    SDParams ParseRequestData(const std::string& requestBody) {
        SDParams params;

        rapidjson::Document document;
        document.Parse(requestBody.c_str());

        if (document.HasParseError()) {
            throw std::runtime_error("Failed to parse JSON request body");
        }

        if (!document.IsObject()) {
            throw std::runtime_error("Invalid JSON request body");
        }

        if (document.HasMember("mode") && document["mode"].IsString()) {
            std::string mode = document["mode"].GetString();
            if (mode == "txt2img") {
                params.mode = TXT2IMG;
            } else if (mode == "img2img") {
                params.mode = IMG2IMG;
            } else if (mode == "img2vid") {
                params.mode = IMG2VID;
            } else if (mode == "convert") {
                params.mode = CONVERT;
            } else {
                throw std::runtime_error("Invalid mode specified");
            }
        } else {
            throw std::runtime_error("Mode not specified");
        }

        if (document.HasMember("prompt") && document["prompt"].IsString()) {
            params.prompt = document["prompt"].GetString();
        } else {
            throw std::runtime_error("Prompt not specified");
        }

        if (document.HasMember("negative_prompt") && document["negative_prompt"].IsString()) {
            params.negative_prompt = document["negative_prompt"].GetString();
        }

        if (document.HasMember("width") && document["width"].IsInt()) {
            params.width = document["width"].GetInt();
        }

        if (document.HasMember("height") && document["height"].IsInt()) {
            params.height = document["height"].GetInt();
        }

        if (document.HasMember("num_inference_steps") && document["num_inference_steps"].IsInt()) {
            params.sample_steps = document["num_inference_steps"].GetInt();
        }

        if (document.HasMember("guidance_scale") && document["guidance_scale"].IsNumber()) {
            params.cfg_scale = static_cast<float>(document["guidance_scale"].GetDouble());
        }

        if (document.HasMember("seed") && document["seed"].IsInt64()) {
            params.seed = document["seed"].GetInt64();
        }

        if (document.HasMember("batch_count") && document["batch_count"].IsInt()) {
            params.batch_count = document["batch_count"].GetInt();
        }

        if (document.HasMember("input_path") && document["input_path"].IsString()) {
            params.input_path = document["input_path"].GetString();
        }

        if (document.HasMember("control_image_path") && document["control_image_path"].IsString()) {
            params.control_image_path = document["control_image_path"].GetString();
        }

        if (document.HasMember("strength") && document["strength"].IsNumber()) {
            params.strength = static_cast<float>(document["strength"].GetDouble());
        }

        if (document.HasMember("control_strength") && document["control_strength"].IsNumber()) {
            params.control_strength = static_cast<float>(document["control_strength"].GetDouble());
        }

        if (document.HasMember("video_frames") && document["video_frames"].IsInt()) {
            params.video_frames = document["video_frames"].GetInt();
        }

        if (document.HasMember("fps") && document["fps"].IsInt()) {
            params.fps = document["fps"].GetInt();
        }

        if (document.HasMember("motion_bucket_id") && document["motion_bucket_id"].IsInt()) {
            params.motion_bucket_id = document["motion_bucket_id"].GetInt();
        }

        if (document.HasMember("n_threads") && document["n_threads"].IsInt()) {
            params.n_threads = document["n_threads"].GetInt();
        } else {
            params.n_threads = m_numThreads;
        }

        return params;
    }

    sd_image_t* GenerateImage(const SDParams& params) {
   sd_image_t* result = nullptr;
   sd_image_t* control_image = nullptr;

   if (params.mode == TXT2IMG) {
       result = txt2img(m_sdContext, params.prompt.c_str(), params.negative_prompt.c_str(), params.clip_skip,
                        params.cfg_scale, params.width, params.height, params.sample_method, params.sample_steps,
                        params.strength, params.seed, params.batch_count, nullptr, params.control_strength,
                        params.style_ratio, params.normalize_input, params.input_id_images_path.c_str());
   } else if (params.mode == IMG2IMG) {
       int width, height, channels;
       uint8_t* input_image_buffer = stbi_load(params.input_path.c_str(), &width, &height, &channels, 3);
       if (input_image_buffer) {
           sd_image_t input_image = {(uint32_t)width, (uint32_t)height, 3, input_image_buffer};
           result = img2img(m_sdContext, &input_image, params.prompt.c_str(), params.negative_prompt.c_str(),
                            params.clip_skip, params.cfg_scale, params.width, params.height, params.sample_method,
                            params.sample_steps, params.strength, params.seed, params.batch_count, nullptr,
                            params.control_strength, params.style_ratio, params.normalize_input, params.input_id_images_path.c_str());
           free(input_image_buffer);
       } else {
           throw std::runtime_error("Failed to load input image");
       }
   } else if (params.mode == IMG2VID) {
       int width, height, channels;
       uint8_t* input_image_buffer = stbi_load(params.input_path.c_str(), &width, &height, &channels, 3);
       if (input_image_buffer) {
           sd_image_t input_image = {(uint32_t)width, (uint32_t)height, 3, input_image_buffer};
           result = img2vid(m_sdContext, &input_image, params.prompt.c_str(), params.negative_prompt.c_str(),
                            params.width, params.height, params.min_cfg, params.cfg_scale, params.sample_method,
                            params.sample_steps, params.strength, params.seed, params.video_frames,
                            params.motion_bucket_id, params.fps, params.augmentation_level);
           free(input_image_buffer);
       } else {
           throw std::runtime_error("Failed to load input image");
       }
   } else if (params.mode == CONVERT) {
       bool success = convert(params.model_path.c_str(), params.vae_path.c_str(), params.output_path.c_str(), params.wtype);
       if (!success) {
           throw std::runtime_error("Failed to convert model");
       }
   }

   if (params.controlnet_path.size() > 0 && params.control_image_path.size() > 0) {
       int width, height, channels;
       uint8_t* control_image_buffer = stbi_load(params.control_image_path.c_str(), &width, &height, &channels, 3);
       if (control_image_buffer) {
           control_image = new sd_image_t{(uint32_t)width, (uint32_t)height, 3, control_image_buffer};
           if (params.canny_preprocess) {
               control_image->data = preprocess_canny(control_image->data, control_image->width, control_image->height,
                                                      0.08f, 0.08f, 0.8f, 1.0f, false);
           }
       }
   }

   if (result && control_image) {
       sd_image_t tmp_result = *result;
       free(result);
       result = txt2img(m_sdContext, params.prompt.c_str(), params.negative_prompt.c_str(), params.clip_skip,
                        params.cfg_scale, params.width, params.height, params.sample_method, params.sample_steps,
                        params.strength, params.seed, params.batch_count, control_image, params.control_strength,
                        params.style_ratio, params.normalize_input, params.input_id_images_path.c_str());
       free(tmp_result.data);
   }

   if (control_image) {
       free(control_image->data);
       free(control_image);
   }

   return result;
}

private:
    std::string m_modelPath;
    std::string m_vaePath;
    int m_numThreads;
    httplib::Server m_server;
    sd_ctx_t* m_sdContext;
};

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_path> <vae_path> <num_threads>" << std::endl;
        return 1;
    }

    std::string modelPath = argv[1];
    std::string vaePath = argv[2];
    int numThreads = std::stoi(argv[3]);

    try {
        StableDiffusionServer server(modelPath, vaePath, numThreads);
        server.Start(8080);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}