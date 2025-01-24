# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import riva.client
import riva.client.audio_io
from dataclasses import dataclass

@dataclass
class Args:
    input_device: int = 1
    list_devices: bool = False
    profanity_filter: bool = False
    automatic_punctuation: bool = True
    no_verbatim_transcripts: bool = False
    language_code: str = "en-US"
    model_name: str = ""
    boosted_lm_words: list = None
    boosted_lm_score: float = 4.0
    speaker_diarization: bool = False
    diarization_max_speakers: int = 3
    start_history: int = -1
    start_threshold: float = -1.0
    stop_history: int = -1
    stop_threshold: float = -1.0
    stop_history_eou: int = -1
    stop_threshold_eou: float = -1.0
    custom_configuration: str = ""
    server: str = "35.239.242.80:50051"
    ssl_cert: str = None
    use_ssl: bool = False
    metadata: dict = None
    sample_rate_hz: int = 16000
    file_streaming_chunk: int = 1600

def main() -> None:
    args = Args()
    print(args)
    auth = riva.client.Auth(args.ssl_cert, args.use_ssl, args.server, args.metadata)
    asr_service = riva.client.ASRService(auth)
    config = riva.client.StreamingRecognitionConfig(
        config=riva.client.RecognitionConfig(
            encoding=riva.client.AudioEncoding.LINEAR_PCM,
            language_code=args.language_code,
            model=args.model_name,
            max_alternatives=1,
            profanity_filter=args.profanity_filter,
            enable_automatic_punctuation=args.automatic_punctuation,
            verbatim_transcripts=not args.no_verbatim_transcripts,
            sample_rate_hertz=args.sample_rate_hz,
            audio_channel_count=1,
        ),
        interim_results=True,
    )
    riva.client.add_word_boosting_to_config(
        config, args.boosted_lm_words, args.boosted_lm_score
    )
    riva.client.add_endpoint_parameters_to_config(
        config,
        args.start_history,
        args.start_threshold,
        args.stop_history,
        args.stop_history_eou,
        args.stop_threshold,
        args.stop_threshold_eou,
    )
    with riva.client.audio_io.MicrophoneStream(
        args.sample_rate_hz,
        args.file_streaming_chunk,
        device=args.input_device,
    ) as audio_chunk_iterator:
        riva.client.print_streaming(
            responses=asr_service.streaming_response_generator(
                audio_chunks=audio_chunk_iterator,
                streaming_config=config,
            ),
            show_intermediate=True,
        )

if __name__ == "__main__":
    main()
