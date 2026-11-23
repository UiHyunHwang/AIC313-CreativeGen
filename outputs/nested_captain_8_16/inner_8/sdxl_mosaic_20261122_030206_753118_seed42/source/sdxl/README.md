# CreativeGen용 SDXL PhotoQuilt

이 디렉터리는 PhotoQuilt의 핵심 절차를 SDXL로 옮긴 로컬 추가 구현입니다.
팀이 작성한 prompt와 seed로 target을 생성하고, 그 latent를 bicubic으로
확대한 뒤 재노이즈하여 고정된 타일 안에서 denoising합니다. 마지막에는
SDXL의 공식 VAE로 전체 latent를 decode합니다. 학습은 필요하지 않습니다.

**현재 상태:** 2026-11-19에 GRID A100D-20C 20GiB vGPU에서 기존 `gs`
작업과 함께 보호 설정을 적용한 2048×2048 생성 한 건을 완료했습니다.
target 1024, 타일 1024(2×2), seed 42, 20단계/타일 tail 12단계,
sequential CPU offload, allocator 상한 8GiB, 최소 여유 3GiB를 사용했습니다.
총 193.22초, PyTorch peak allocated 3.075GiB / reserved 3.102GiB,
감시 중 GPU 최소 여유 10.978GiB였습니다. 이 수치는 CUDA context 등을 포함한
프로세스 전체 메모리의 최대값은 아닙니다. 모의 저메모리 상황에서 새 시험
프로세스만 종료하는 감시 동작도 확인했습니다. 전체 설정 재실행의 비트 단위
재현성, 기본 4096 출력 및 다른 설정은 아직 검증하지 않았습니다.
이번 작업에서 의존성을 설치하지 않고 기존 환경과 캐시된 공식 가중치를 사용했습니다.

추가로 4096×4096 / 1024 타일 4×4 실행도 같은 보호 설정과 20/12단계로 완료했습니다.
전역 prompt는 기존 섬·주택 수채화를 유지하고, 타일 prompt는 Codex가 작성한
새 한 마리 수채화 문장으로 변경했습니다(AI prompt 작성 도움은 write-up에 명시).
578.61초, PyTorch peak allocated 3.331GiB / reserved 3.359GiB,
최소 GPU 여유 11.424GiB였으며 타일 16장을 함께 저장했습니다.
기본값인 50단계/model offload 설정은 이 추가 실행으로 검증한 것이 아닙니다.

8×8과 16×16도 완료했습니다. 두 실행은 seed 42와 기존 전역/새 타일 prompt,
20/12단계, sequential target offload, model tail offload, serial CFG,
CPU tile decode, 8GiB allocator 상한과 3GiB 여유 감시를 사용했습니다.

| 그리드 | 출력 크기 | 총 시간 | Peak allocated / reserved (GiB) | 최소 GPU 여유 (GiB) |
| --- | --- | --- | --- | --- |
| 8×8 | 8192×8192 | 468.42초 | 5.101 / 5.814 | 7.234 |
| 16×16 | 16384×16384 | 1515.44초 | 5.168 / 5.879 | 10.795 |

실행 폴더는 각각 `outputs/sdxl_guarded/sdxl_mosaic_20261119_024325_366394_seed42`,
`outputs/sdxl_guarded/sdxl_mosaic_20261119_025146_189416_seed42`입니다.
원본 PNG와 64/256개 타일, 최종 latent, 설정, 소스 및 환경 기록을 저장했습니다.
각 `config.json`으로 전체 생성 재실행이 가능하지만 이 두 설정의 전체 재생성 및
비트 단위 출력 일치는 아직 별도로 검증하지 않았습니다. GPU 여유 수치는 실행
당시 다른 작업의 점유도 반영하며, 프로세스 전체 메모리 상한을 뜻하지 않습니다.
원본 PNG 크기는 약 86MiB / 303MiB입니다. 큰 원본을 Pillow로 읽을 때는
자체 생성한 파일임을 확인한 뒤 해당 미리보기 프로세스에서
`Image.MAX_IMAGE_PIXELS = None`으로 크기 제한을 해제할 수 있습니다.

## 실행 준비

Python 3.10 이상과 환경에 맞는 PyTorch/CUDA 구성이 필요합니다. 저장소 루트
`PhotoQuilt/`에서 실행합니다. 이 추가 구현의 의존성은
[`requirements-sdxl.txt`](../requirements-sdxl.txt)에 별도로 두었습니다.
GPU에 맞는 PyTorch 빌드를 먼저 선택하고 나머지 패키지를 설치합니다.

```bash
pip install -r requirements-sdxl.txt
```

이 파일의 버전 범위는 검증된 환경을 뜻하지 않습니다. 실제 생성 실행 시
주요 생성 패키지의 정확한 버전은 `requirements-runtime.txt`와
`environment.json`에 남깁니다.

## 생성

`TEAM_AUTHORED_GLOBAL_PROMPT`와 `TEAM_AUTHORED_TILE_PROMPT`는 팀이 직접
작성한 문장으로 교체합니다.

```bash
python -m sdxl \
  --prompt "TEAM_AUTHORED_GLOBAL_PROMPT" \
  --tile-prompt "TEAM_AUTHORED_TILE_PROMPT" \
  --seed 42 \
  --height 4096 --width 4096 \
  --tile-size 1024 \
  --steps 50 --noise-steps 30 \
  --tile-batch 1 \
  --offload model \
  --output-dir outputs/sdxl
```

target 생성과 타일 생성은 모두
`stabilityai/stable-diffusion-xl-base-1.0`의 동일 revision을 사용합니다.
외부 이미지 로딩, refiner, adapter, 추가 VAE, 학습형 upscaler 경로는 없습니다.
타일은 겹치지 않으며 출력 높이와 너비는 `--tile-size`의 배수여야 합니다.
SDXL에 맞춰 dual text encoder, pooled embeddings, 크기 conditioning을 사용하고,
타일마다 같은 DDIM 시간표를 적용합니다. 타일 경계는 모자이크의 일부입니다.

주요 옵션:

| 옵션 | 의미 / 기본값 |
| --- | --- |
| `--bootstrap-size` | target의 긴 변 길이. 기본 `1024`, 종횡비 유지 |
| `--tile-prompt` | 타일의 내용. 생략 시 전역 prompt 사용 |
| `--negative-prompt` | 두 생성 단계의 negative prompt. 기본 빈 문자열 |
| `--target-seed` | target 생성 seed. 기본 `--seed` |
| `--noise-seed` | 확대 latent의 재노이즈 seed. 기본 `(seed + 1) % 2**63` |
| `--noise-steps` | 전체 `--steps` 시간표 중 마지막에 실행할 타일 단계 수 |
| `--guidance-scale` | 두 생성 단계의 classifier-free guidance 강도. 기본 `5.0` |
| `--tile-batch` | 한 UNet 호출에서 처리하는 타일 수. 기본 `1` |
| `--dtype` | `float16` / `bfloat16` / `float32`. 기본 `float16` |
| `--offload` | `none` / `model` / `sequential`. 기본 `model` |
| `--tail-offload` | 타일 구간만 `model` 또는 `sequential`로 전환. 생략 시 `--offload` 유지 |
| `--tail-serial-cfg` | 타일의 조건부·무조건부 UNet 계산을 따로 수행해 activation 메모리 절약 |
| `--decode-mode cpu-tiles` | VAE 복원 타일과 전체 픽셀 canvas를 CPU에 저장. 기본은 `standard` |
| `--device` | `cuda` / `cpu`. 기본 `cuda` |
| `--attention-slicing` | 메모리 절약용 attention slicing 사용 |
| `--no-deterministic` | 기본 deterministic 설정 해제 |
| `--revision` | 사용할 Hugging Face revision. 실행 시 immutable commit으로 고정 |
| `--local-files-only` | 이미 캐시된 모델 파일만 사용 |
| `--save-tiles` | 완성된 mosaic의 타일별 crop도 저장 |
| `--gpu-memory-limit-gib` | 이 생성 프로세스의 PyTorch CUDA allocator 상한(GiB). 기본 제한 없음 |
| `--min-free-gpu-gib` | GPU 전체 여유가 이 값 미만이면 이 생성 프로세스만 중단. 기본 감시 꺼짐 |

CPU로 실행하려면 `--device cpu --dtype float32 --offload none`을 함께 지정합니다.

`--noise-steps / --steps`는 남은 denoising 단계의 비율입니다. SDXL에서는
DDIM의 noise schedule로 재노이즈하므로 FLUX 논문의 선형 noise 혼합 계수와
동일한 값으로 해석하면 안 됩니다. 타일 batch와 offload는 메모리·속도에 영향을
주며, 다른 설정에서의 재현 결과는 별도로 검증해야 합니다.

`--local-files-only`는 네트워크 없이 캐시만 사용합니다. `main` 참조가 없더라도
공식 모델의 캐시 snapshot이 하나이면 그 commit을 선택하고 기록합니다.
여러 snapshot이 있으면 `--revision`으로 지정해야 합니다. 기존 시험에서 사용한
revision은 `462165984030d82259a11f4367a4eed129e94a7b`입니다.

다른 GPU 작업과 동시에 작은 시험을 할 때의 설정 예시:

```bash
python -m sdxl \
  --prompt "TEAM_AUTHORED_GLOBAL_PROMPT" \
  --seed 42 --height 2048 --width 2048 \
  --bootstrap-size 1024 --tile-size 1024 \
  --steps 20 --noise-steps 12 --tile-batch 1 \
  --offload sequential \
  --gpu-memory-limit-gib 8 --min-free-gpu-gib 3 \
  --output-dir outputs/sdxl_guarded
```

상한은 PyTorch allocator에 적용되며 CUDA context 등 모든 GPU 메모리를 제한하지는
않습니다. 별도 CPU 감시 프로세스가 0.5초 간격으로 여유 메모리를 조회하고, 기준
미달 또는 조회 실패 시 자신이 시작된 생성 프로세스만 종료합니다. PID와 시작 시각을
확인하므로 다른 작업에 종료 신호를 보내지 않습니다. 감시 내역은 `memory_guard.json`에
기록합니다. 급격한 동시 할당을 완전히 막거나 다른 작업의 메모리를 예약해 주는
기능은 아닙니다. 동시 실행 시 연산 속도와 CPU 자원도 서로 영향을 받습니다.

큰 canvas에서는 `--offload sequential --tail-offload model --tail-serial-cfg
--decode-mode cpu-tiles` 조합을 사용할 수 있습니다. target은 순차 offload로 생성하고,
타일 구간에서는 UNet을 GPU에 유지하여 전송 비용을 줄입니다. CFG는 같은 공식을
두 번의 작은 호출로 계산하며, batch 처리와의 수치 반올림 차이는 발생할 수 있습니다.
CPU decode는 공식 VAE의 겹침·blending 순서를 유지하면서 복원된 픽셀을 host 메모리로
옮깁니다. GPU 메모리를 줄이는 대신 CPU 메모리는 출력 면적에 따라 증가합니다.
이 옵션 역시 `--gpu-memory-limit-gib`와 `--min-free-gpu-gib` 아래에서 실행합니다.
타일 denoising 완료 직후 `final_latents.safetensors`를 저장하므로 decode 이전의
계산 결과도 보관됩니다. 입력 이미지를 외부에서 가져오는 기능은 추가하지 않았습니다.

## 저장된 설정으로 재생성

```bash
python -m sdxl \
  --config outputs/sdxl/RUN/config.json \
  --output-dir outputs/reproduced
```

`RUN`은 실제 실행 디렉터리 이름으로 교체합니다. `--config`와 생성 옵션을
함께 지정하지 않습니다. 이 명령은 저장된 PNG를 입력으로 읽지 않고,
기록된 prompt·seed·checkpoint revision·전체 설정에서 target과 mosaic을 다시
생성합니다. 최초 실행에서 기록한 소스와 환경도 함께 보관해야 합니다.
저장소 코드가 바뀌었다면 원래 실행의 `source/`를 작업 디렉터리로 삼아 저장된
코드를 사용합니다. `--config`에는 원래 `config.json`의 경로를 지정합니다.

```bash
cd "/absolute/path/to/RUN/source"
python -m sdxl --config ../config.json --output-dir ../../reproduced
```

실행 디렉터리의 주요 산출물:

| 파일 | 내용 |
| --- | --- |
| `target.png` | 파이프라인이 생성한 저해상도 target |
| `mosaic.png` | 최종 모자이크 |
| `requested_config.json` | 모델 로딩 전에 저장한 요청 설정 |
| `config.json` | 재실행용 생성 설정과 고정된 모델 revision |
| `manifest.json` | 실행 및 산출물 기록 |
| `environment.json` | 하드웨어·라이브러리 등 환경 정보 |
| `requirements-runtime.txt` | 실행 당시 주요 생성 패키지의 정확한 버전 |
| `source/` | 해당 실행에 사용한 소스 스냅샷 |
| `tiles/` | `--save-tiles` 사용 시 최종 결과의 타일별 crop |

`manifest.json`에는 완료·실패 상태, seed로 생성한 target/mosaic 수, 단계별 시간,
GPU 실행 시 peak 메모리, 산출물의 SHA256이 기록됩니다. 원래 실행의 manifest가
남아 있으면 재생성 결과의 SHA256과 소스·주요 패키지 버전의 일치 여부도
기록합니다. 일치 여부는 실행 후 확인되며, 환경 차이를 자동으로 해결하지는
않습니다.

기본 deterministic 설정은 같은 소스·checkpoint·환경에서의 재현을 목표로
합니다. PyTorch/CUDA, GPU, 정밀도, attention 구현 등이 달라지면 bit 단위로
동일한 결과를 보장할 수 없습니다. 제출 전에는 실제 KCLOUD 단일 20GB GPU에서
최종 설정 전체를 검증해야 합니다.

## 타일 안에 두 번째 모자이크 생성하기

안쪽 격자의 `parent_tile_source`에는 바깥 실행 경로(`run`), 그 실행의
`final_latents.safetensors` SHA-256(`sha256`), 0부터 세는 `row`와
`column`을 기록합니다. 안쪽 실행은 해시와 SDXL revision을 확인한 뒤 해당
타일의 latent를 읽습니다. 바깥 타일의 한 변과 안쪽 `bootstrap_size`를
같게 맞춥니다. 바깥 실행을 처음부터 재현하려면 먼저 해당 실행의
`config.json`으로 아이언맨과 스파이더맨 256칸을 생성합니다.

아이언맨 실험의 `outputs/nested_captain_8_16/selection.json`에는 같은
CLIP 모델로 평가한 바깥 타일 256개의 점수와 좌표가 들어 있습니다.
`inner_8/recipe.json`과 `inner_16/recipe.json`은 동일한 스파이더맨
타일에서 시작하고 안쪽 프롬프트로 `captain america`를 사용합니다.
저장소 루트에서 다음과 같이 두 크기를 개별 재현할 수 있습니다.

```bash
experiment=outputs/nested_captain_8_16
for grid in 8 16; do
  OMP_NUM_THREADS=2 python -m sdxl --local-files-only \
    --config "$experiment/inner_$grid/recipe.json" \
    --output-dir "outputs/reproduced/inner_$grid"
done
```

바깥 실행을 새 경로에 재생성했으면 `python "$experiment/select_tile.py"
--parent-run <새로운_실행_경로>`를 한 줄로 실행하고 `PYTHONPATH=.
python "$experiment/recipes.py"`로 두 설정을 갱신합니다. 기존에 생성한
이미지를 확인할 때는 `zoom.html`을 브라우저에서 열고 안쪽 격자를 선택합니다.

## 과제 기록과 출처

전체 실험의 생성 수, 최종 sample 선택 기준, 선택한 seed 및 AI로 prompt를
구상했는지는 write-up에 별도로 기록합니다. 개별 실행 디렉터리만으로 전체
실험 횟수나 선택 기준이 자동 집계되지는 않습니다. 사용한 방법·코드·모델과
AI 코딩 도움도 write-up에 명시합니다.

참고한 방법·코드·모델:

- [PhotoQuilt 논문 v2](https://arxiv.org/abs/2606.30968v2) 및
  [공식 저장소](https://github.com/KooroshRH/PhotoQuilt)
- [SDXL 논문](https://arxiv.org/abs/2307.01952) 및
  [허용된 SDXL base checkpoint](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [DDIM 논문](https://arxiv.org/abs/2010.02502)
- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)

이 구현의 결과는 SDXL 이식 결과이며, 논문의 FLUX·SD2.1 실험 수치를 그대로
재현했다고 주장하지 않습니다.

### 타일 시작 시점과 2×2 유도 실험

`--steps 20 --noise-steps 12`에서 `--tile-schedule-steps 13`을 추가하면 target은 20단계로 생성하고 타일은 DDIM 13단계 시간표의 마지막 12개 시점을 사용한다. 현재 모델의 leading 시간표에서 첫 타일 시점은 836이다. 이 옵션을 생략한 기존 설정에서는 550이다.

순수 노이즈에서 타일을 시작하려면 `--tile-schedule-steps 12 --noise-steps 12 --tile-palette-guidance 0.25`처럼 설정한다. 이 실험적 변형은 처음 4번의 업데이트로 타일 자체의 구도를 잡고, 남은 8번 동안 예측한 깨끗한 latent와 target 사이의 채널별 2×2 블록 평균 차이를 보간해 DDIM 업데이트에 더한다. 강도는 업데이트마다 적용된다. 이 값이 커질수록 큰 색면이 target을 더 강하게 따른다. 두 모드 모두 추가 학습이나 외부 이미지 없이 실행할 수 있다.

### 일부 타일의 seed 재선택

`tile_noise_seeds`는 위치별 타일 seed를 담는 선택적 JSON 객체다. 예를 들어 `{"r000_c001": 12345}`는 첫 행 두 번째 타일의 시작 노이즈를 독립적인 CPU 난수로 생성한다. 먼저 원래 캔버스 전체의 노이즈를 생성하므로 다른 위치의 난수는 같은 레시피에서 재현된다. 선택 결과는 출력 `config.json`에 기록되며 `python -m sdxl --config ... --local-files-only`로 target과 전체 모자이크를 다시 만들 수 있다.
