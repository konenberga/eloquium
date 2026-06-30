# Reference voice assets

`ref_ru.wav` — the default Russian reference voice F5-TTS imitates for
`language: "ru"`. A native Russian reference is essential: the bundled English
default voice injects a foreign accent into Russian output.

- Source: [google/fleurs](https://huggingface.co/datasets/google/fleurs)
  `ru_ru` split, licensed **CC BY 4.0**.
- Transcript (set as `F5_REF_TEXT` in docker-compose.yml):
  «как только вы выйдете из течения плыть обратно будет не труднее чем обычно»

Swap in your own voice by replacing this file and updating `F5_REF_TEXT`
(see CLAUDE.md → Environment Variables).
