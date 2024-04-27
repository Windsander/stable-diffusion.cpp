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

struct LaunchParams {
    int n_threads = -1;
    std::string model_path;
    std::string vae_path;
    std::string taesd_path;
    std::string controlnet_path;
    std::string embeddings_path;
    std::string stacked_id_embeddings_path;
    std::string lora_model_dir;
    std::string host = "127.0.0.1";
    schedule_t schedule = DEFAULT;
    rng_type_t rng_type = STD_DEFAULT_RNG; //CUDA_RNG;

    sd_type_t wtype = SD_TYPE_COUNT;
    bool vae_tiling = false;
    bool control_net_cpu = false;
    bool clip_on_cpu = false;
    bool vae_on_cpu = false;

    int port = 8080;
};

struct RequestParams {
    SDMode mode = TXT2IMG;
    std::string input_id_images_path;
    std::string input_image_data;
    std::string control_image_data;
    std::string prompt;
    std::string negative_prompt;
    std::string output_format = "png";
    
    float output_quality = 0.9;

    float min_cfg = 1.0f;
    float cfg_scale = 7.0f;
    float style_ratio = 20.f;
    int clip_skip = -1;
    int width = 512;
    int height = 512;
    int batch_count = 1;
    int video_frames = 6;
    int motion_bucket_id = 127;
    int fps = 6;
    float augmentation_level = 0.f;
    sample_method_t sample_method = EULER_A;

    int sample_steps = 20;
    float strength = 0.75f;
    float control_strength = 0.9f;
    
    int64_t seed = -1;
    bool normalize_input = false;
    bool canny_preprocess = false;
    int upscale_repeats = 1;
};

void print_launch_params(const LaunchParams& params) {
    printf("Launch Options:\n");
    printf("    n_threads:         %d\n", params.n_threads);
    printf("    host:              %s\n", params.host.c_str());
    printf("    port:              %d\n", params.port);
    printf("    model_path:        %s\n", params.model_path.c_str());
    printf("    wtype:             %s\n", params.wtype < SD_TYPE_COUNT ? sd_type_name(params.wtype) : "unspecified");
    printf("    vae_path:          %s\n", params.vae_path.c_str());
    printf("    taesd_path:        %s\n", params.taesd_path.c_str());
    printf("    controlnet_path:   %s\n", params.controlnet_path.c_str());
    printf("    embeddings_path:   %s\n", params.embeddings_path.c_str());
    printf("    stacked_id_embeddings_path:   %s\n", params.stacked_id_embeddings_path.c_str());
    printf("    lora_model_dir:    %s\n", params.lora_model_dir.c_str());

    printf("    rng_type:          %s\n", rng_type_to_str[params.rng_type]);
    printf("    schedule:          %s\n", schedule_str[params.schedule]);

    printf("    clip on cpu:       %s\n", params.clip_on_cpu ? "true" : "false");
    printf("    controlnet cpu:    %s\n", params.control_net_cpu ? "true" : "false");
    printf("    vae decoder on cpu:%s\n", params.vae_on_cpu ? "true" : "false");
    printf("    vae_tiling:        %s\n", params.vae_tiling ? "true" : "false");
}

void print_request_params(const RequestParams& params) {
    printf("Request Options:\n");
    printf("    mode:              %s\n", modes_str[params.mode]);
    printf("    input_id_images_path:   %s\n", params.input_id_images_path.c_str());
    printf("    style ratio:       %.2f\n", params.style_ratio);
    printf("    normalize input image:  %s\n", params.normalize_input ? "true" : "false");
    printf("    input_image_data:   %s\n", params.input_image_data.substr(0, std::min(params.input_image_data.size(), size_t(16))).c_str());
    printf("    control_image_data: %s\n", params.control_image_data.substr(0, std::min(params.control_image_data.size(), size_t(16))).c_str());
    printf("    strength(control): %.2f\n", params.control_strength);
    printf("    prompt:            %s\n", params.prompt.c_str());
    printf("    negative_prompt:   %s\n", params.negative_prompt.c_str());
    printf("    min_cfg:           %.2f\n", params.min_cfg);
    printf("    cfg_scale:         %.2f\n", params.cfg_scale);
    printf("    clip_skip:         %d\n", params.clip_skip);
    printf("    width:             %d\n", params.width);
    printf("    height:            %d\n", params.height);
    printf("    sample_method:     %s\n", sample_method_str[params.sample_method]);
    printf("    sample_steps:      %d\n", params.sample_steps);
    printf("    strength(img2img): %.2f\n", params.strength);
    printf("    seed:              %lld\n", params.seed);
    printf("    video-total-frames: %d\n", params.video_frames);
    printf("    video-fps:         %d\n", params.fps);
    printf("    motion-bucket-id:  %d\n", params.motion_bucket_id);
    printf("    batch_count:       %d\n", params.batch_count);
    printf("    upscale_repeats:   %d\n", params.upscale_repeats);
    printf("    canny_preprocess:  %s\n", params.canny_preprocess ? "true" : "false");
    printf("    output_format:     %s\n", params.output_format.c_str());
    printf("    output_quality:    %.2f\n", params.output_quality);
}

std::string base64_decode(const std::string &in) {
    static const auto inverse_lookup = []() {
        std::array<int, 256> inv;
        inv.fill(-1);
        for (size_t i = 0; i < 64; i++) {
            inv[static_cast<uint8_t>("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i])] = i;
        }
        return inv;
    }();

    std::string out;
    out.reserve(in.size() * 3 / 4);

    auto val = 0;
    auto valb = -8;

    for (auto c : in) {
        if (c == '=') {
            break;
        }
        auto value = inverse_lookup[static_cast<uint8_t>(c)];
        if (value == -1) {
            throw std::runtime_error("Invalid base64 character");
        }
        val = (val << 6) + value;
        valb += 6;
        if (valb >= 0) {
            out.push_back(static_cast<char>((val >> valb) & 0xFF));
            valb -= 8;
        }
    }

    return out;
}

class StableDiffusionServer {
public:
    StableDiffusionServer(const LaunchParams& params)
        : m_launchParams(params) {}

    void Start() {
        InitializeModel();
        SetupRoutes();
        m_server.listen(m_launchParams.host.c_str(), m_launchParams.port);
    }

private:
    void InitializeModel() {
        m_sdContext = new_sd_ctx(m_launchParams.model_path.c_str(),
                                  m_launchParams.vae_path.c_str(),
                                  m_launchParams.taesd_path.c_str(),
                                  m_launchParams.controlnet_path.c_str(),
                                  m_launchParams.lora_model_dir.c_str(),
                                  m_launchParams.embeddings_path.c_str(),
                                  m_launchParams.stacked_id_embeddings_path.c_str(),
                                  false,
                                  m_launchParams.vae_tiling,
                                  true,
                                  m_launchParams.n_threads,
                                  m_launchParams.wtype,
                                  m_launchParams.rng_type,
                                  m_launchParams.schedule,
                                  m_launchParams.clip_on_cpu,
                                  m_launchParams.control_net_cpu,
                                  m_launchParams.vae_on_cpu);
        if (!m_sdContext) {
            throw std::runtime_error("Failed to initialize Stable Diffusion model");
        }
    }

   

    void SendImageResponse(httplib::Response& res, const sd_image_t* image, const std::string& format, float quality = 0.9) {
        res.status = 200;
        res.set_header("Content-Type", "image/" + format);

        if (format == "png") {
            stbi_write_png_to_func(WriteImageToResponse, &res, image->width, image->height, image->channel, image->data, 0);
        } else if (format == "jpg" || format == "jpeg") {
            stbi_write_jpg_to_func(WriteImageToResponse, &res, image->width, image->height, image->channel, image->data, int(quality * 100));
        } else {
            throw std::runtime_error("Unsupported image format");
        }
    }

    void SetupRoutes() {
        m_server.Post("/generate", [this](const httplib::Request& req, httplib::Response& res) {
            try {
                RequestParams params = ParseRequestData(req.body);
                print_request_params(params);
                sd_image_t* result = GenerateImage(params);
                if (result) {
                    SendImageResponse(res, result, params.output_format);
                    //res.set_content((const char*)result->data, result->width * result->height * result->channel, "image/png");
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

    RequestParams ParseRequestData(const std::string& requestBody) {
        RequestParams params;

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
            } else {
                throw std::runtime_error("Invalid mode specified");
            }
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

        if (document.HasMember("input_image") && document["input_image"].IsString()) {
            params.input_image_data = document["input_image"].GetString();
        }

        if (document.HasMember("control_image") && document["control_image"].IsString()) {
            params.control_image_data = document["control_image"].GetString();
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

        //format
        if (document.HasMember("output_format") && document["output_format"].IsString()) {
            params.output_format = document["output_format"].GetString();
            if (params.output_format != "png" && params.output_format != "jpg" && params.output_format != "jpeg") {
                throw std::runtime_error("Invalid output format specified");
            }
        }

        //quality
        if (document.HasMember("output_quality") && document["output_quality"].IsNumber()) {
            params.output_quality = static_cast<float>(document["output_quality"].GetDouble());
            if (params.output_quality < 0.0 || params.output_quality > 1.0) {
                throw std::runtime_error("Invalid output quality specified");
            }
        }
        return params;
    }

    sd_image_t* GenerateImage(const RequestParams& params) {
      sd_image_t* result = nullptr;
      sd_image_t* control_image = nullptr;

      if (params.mode == TXT2IMG) {
          result = txt2img(m_sdContext, params.prompt.c_str(), params.negative_prompt.c_str(), params.clip_skip,
                            params.cfg_scale, params.width, params.height, params.sample_method, params.sample_steps,
                            params.strength, params.seed, params.batch_count, nullptr, params.control_strength,
                            params.style_ratio, params.normalize_input, params.input_id_images_path.c_str());
      } else if (params.mode == IMG2IMG) {
          int width, height, channels;

          sd_image_t* inputImage = DataURIToSDImage(params.input_image_data);
          if (inputImage) {
              result = img2img(m_sdContext, inputImage, params.prompt.c_str(), params.negative_prompt.c_str(),
                                params.clip_skip, params.cfg_scale, params.width, params.height, params.sample_method,
                                params.sample_steps, params.strength, params.seed, params.batch_count, nullptr,
                                params.control_strength, params.style_ratio, params.normalize_input, params.input_id_images_path.c_str());
              free(inputImage->data);
              delete inputImage;
          } else {
              throw std::runtime_error("Failed to load input image");
          }
      } else if (params.mode == IMG2VID) {
          int width, height, channels;
          sd_image_t* inputImage = DataURIToSDImage(params.input_image_data);
          if (inputImage) {
              result = img2vid(m_sdContext, inputImage, params.prompt.c_str(), params.negative_prompt.c_str(),
                                params.width, params.height, params.min_cfg, params.cfg_scale, params.sample_method,
                                params.sample_steps, params.strength, params.seed, params.video_frames,
                                params.motion_bucket_id, params.fps, params.augmentation_level);
              free(inputImage->data);
              delete inputImage;
          } else {
              throw std::runtime_error("Failed to load input image");
          }
      } else {
          throw std::runtime_error("Invalid mode specified");
      }

      if (m_launchParams.controlnet_path.size() > 0 && params.control_image_data.size() > 0) {
          control_image = DataURIToSDImage(params.control_image_data);
          if (control_image) {
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
          delete control_image;
      }

      return result;
    }


     static void WriteImageToResponse(void* context, void* data, int size) {
        httplib::Response* res = static_cast<httplib::Response*>(context);
        res->body.append(static_cast<char*>(data), size);
    }

    sd_image_t* DataURIToSDImage(const std::string& dataURI) {
        // Extract the base64-encoded data from the data URI
        size_t commaPos = dataURI.find(',');
        if (commaPos == std::string::npos) {
            throw std::runtime_error("Invalid data URI format");
        }
        std::string base64Data = dataURI.substr(commaPos + 1);

        // Decode the base64 data
        std::string decodedData = base64_decode(base64Data);

        // Load the image data into an sd_image_t object
        int width, height, channels;
        uint8_t* imageData = stbi_load_from_memory(reinterpret_cast<const stbi_uc*>(decodedData.data()),
                                                  decodedData.size(), &width, &height, &channels, 3);
        if (!imageData) {
            throw std::runtime_error("Failed to load image from data URI");
        }

        sd_image_t* sdImage = new sd_image_t{(uint32_t)width, (uint32_t)height, 3, imageData};
        return sdImage;
    }

private:
    LaunchParams m_launchParams;
    httplib::Server m_server;
    sd_ctx_t* m_sdContext;
};


void parse_launch_args(int argc, const char** argv, LaunchParams& params) {
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "-t" || arg == "--threads") {
        if (i + 1 < argc) {
            params.n_threads = std::stoi(argv[i + 1]);
            i++;
        } else {
            fprintf(stderr, "Error: --threads requires an argument\n");
        }
    } else if (arg == "-m" || arg == "--model") {
        if (i + 1 < argc) {
            params.model_path = argv[i + 1];
            i++;
        } else {
            fprintf(stderr, "Error: --model requires an argument\n");
        }
    } else if (arg == "--vae") {
        if (i + 1 < argc) {
            params.vae_path = argv[i + 1];
            i++;
        } else {
            fprintf(stderr, "Error: --vae requires an argument\n");
        }
    } else if (arg == "--taesd") {
        if (i + 1 < argc) {
            params.taesd_path = argv[i + 1];
            i++;
        } else {
            fprintf(stderr, "Error: --taesd requires an argument\n");
        }
    } else if (arg == "--control-net") {
        if (i + 1 < argc) {
            params.controlnet_path = argv[i + 1];
            i++;
        } else {
            fprintf(stderr, "Error: --control-net requires an argument\n");
        }
    } else if (arg == "--embd-dir") {
        if (i + 1 < argc) {
            params.embeddings_path = argv[i + 1];
            i++;
        } else {
            fprintf(stderr, "Error: --embd-dir requires an argument\n");
        }
    } else if (arg == "--stacked-id-embd-dir") {
        if (i + 1 < argc) {
            params.stacked_id_embeddings_path = argv[i + 1];
            i++;
        } else {
            fprintf(stderr, "Error: --stacked-id-embd-dir requires an argument\n");
        }
    } else if (arg == "--lora-model-dir") {
        if (i + 1 < argc) {
            params.lora_model_dir = argv[i + 1];
            i++;
        } else {
            fprintf(stderr, "Error: --lora-model-dir requires an argument\n");
        }
    } else if (arg == "--type") {
        if (i + 1 < argc) {
            std::string type = argv[i + 1];
            if (type == "f32") {
                params.wtype = SD_TYPE_F32;
            } else if (type == "f16") {
                params.wtype = SD_TYPE_F16;
            } else if (type == "q4_0") {
                params.wtype = SD_TYPE_Q4_0;
            } else if (type == "q4_1") {
                params.wtype = SD_TYPE_Q4_1;
            } else if (type == "q5_0") {
                params.wtype = SD_TYPE_Q5_0;
            } else if (type == "q5_1") {
                params.wtype = SD_TYPE_Q5_1;
            } else if (type == "q8_0") {
                params.wtype = SD_TYPE_Q8_0;
            } else {
                fprintf(stderr, "Error: invalid weight format %s, must be one of [f32, f16, q4_0, q4_1, q5_0, q5_1, q8_0]\n",
                        type.c_str());
            }
            i++;
        } else {
            fprintf(stderr, "Error: --type requires an argument\n");
        }
    } else if (arg == "--vae-tiling") {
        params.vae_tiling = true;
    } else if (arg == "--control-net-cpu") {
        params.control_net_cpu = true;
    } else if (arg == "--clip-on-cpu") {
        params.clip_on_cpu = true;
    } else if (arg == "--vae-on-cpu") {
        params.vae_on_cpu = true;
    } else if (arg == "--host") {
        if (i + 1 < argc) {
            params.host = argv[i + 1];
            i++;
        } else {
            fprintf(stderr, "Error: --host requires an argument\n");
        }
    } else if (arg == "--port") {
        if (i + 1 < argc) {
            params.port = std::stoi(argv[i + 1]);
            i++;
        } else {
            fprintf(stderr, "Error: --port requires an argument\n");
        }
    } else if (arg == "--schedule") {
      if (i + 1 < argc) {
          const char* schedule_selected = argv[i + 1];
          int schedule_found = -1;
          const int n_schedules = sizeof(schedule_str) / sizeof(schedule_str[0]);
          for (int d = 0; d < n_schedules; d++) {
              if (!strcmp(schedule_selected, schedule_str[d])) {
                  schedule_found = d;
              }
          }
          if (schedule_found == -1) {
              fprintf(stderr, "Error: invalid schedule %s\n", schedule_selected);
          } else {
              params.schedule = (schedule_t)schedule_found;
          }
          i++;
      } else {
          fprintf(stderr, "Error: --schedule requires an argument\n");
      }
    } else if (arg == "--rng") {
      if (i + 1 < argc) {
          const char* rng_selected = argv[i + 1];
          int rng_found = -1;
          const int n_rng_types = sizeof(rng_type_to_str) / sizeof(rng_type_to_str[0]);
          for (int d = 0; d < n_rng_types; d++) {
              if (!strcmp(rng_selected, rng_type_to_str[d])) {
                  rng_found = d;
              }
          }
          if (rng_found == -1) {
              fprintf(stderr, "Error: invalid rng %s\n", rng_selected);
          } else {
              params.rng_type = (rng_type_t)rng_found;
          }
          i++;
      } else {
          fprintf(stderr, "Error: --rng requires an argument\n");
      }
    }
  }

}

int main(int argc, char* argv[]) {
    LaunchParams launchParams;
    parse_launch_args(argc, (const char**)argv, launchParams);
    
    if (launchParams.n_threads <= 0) {
        launchParams.n_threads = get_num_physical_cores();
    }
    print_launch_params(launchParams);
    if (launchParams.model_path.empty()) {
        std::cerr << "Error: Model path is required." << std::endl;
        return 1;
    }

    try {
        StableDiffusionServer server(launchParams);
        server.Start();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}