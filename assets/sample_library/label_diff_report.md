# Sample Label Diff Report

- labeled samples: `146`
- field diff counts: `{"speaker_pattern": 91, "voice_age": 104, "music_pattern": 85, "language": 36}`

## Top diff patterns

- `95` x `voice_age`: `` -> `adult`
- `65` x `music_pattern`: `bgm` -> `no_music`
- `27` x `speaker_pattern`: `male_female_mixed` -> `single_male`
- `21` x `speaker_pattern`: `male_female_mixed` -> `single_female`
- `18` x `speaker_pattern`: `multi_speaker_other` -> `single_female`
- `16` x `language`: `mixed` -> `en`
- `14` x `language`: `en` -> `zh`
- `10` x `speaker_pattern`: `multi_speaker_other` -> `single_male`
- `7` x `music_pattern`: `bgm` -> `transient_sfx`
- `6` x `music_pattern`: `bgm` -> `song`
- `6` x `speaker_pattern`: `multi_speaker_other` -> `male_female_mixed`
- `5` x `music_pattern`: `transient_sfx` -> `no_music`
- `5` x `voice_age`: `adult` -> `child`
- `4` x `language`: `mixed` -> `zh`
- `2` x `speaker_pattern`: `single_female` -> `single_male`
- `2` x `voice_age`: `` -> `unknown`
- `2` x `language`: `` -> `zh`
- `2` x `speaker_pattern`: `` -> `single_female`
- `2` x `music_pattern`: `` -> `no_music`
- `2` x `speaker_pattern`: `single_male` -> `single_female`
- `2` x `speaker_pattern`: `single_male` -> `male_female_mixed`
- `1` x `voice_age`: `` -> `child`
- `1` x `speaker_pattern`: `male_female_mixed` -> `multi_speaker_other`
- `1` x `voice_age`: `adult` -> `unknown`

## Single male mispredictions

- `page_301962_256bfc0f7be011f0b62218c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301993_4555f48f765a11ef996918c04d251e75.mp3`: speaker `single_female`, music `no_music` -> actual `single_male/no_music`
- `page_302000_e787406e637c11ef871d18c04d251e75.mp3`: speaker `multi_speaker_other`, music `transient_sfx` -> actual `single_male/no_music`
- `page_301992_01484076.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301981_03a5b1b075ac11ef98ab18c04d251e75.mp3`: speaker `single_female`, music `no_music` -> actual `single_male/no_music`
- `page_301992_01484074.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301992_01484073.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_357462_0f5edc51780511f08660b32a8ac171c2.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_357462_0eb296c0780511f0a0bbb32a8ac171c2.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_302000_7416e130648111ef820418c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_302000_73e263ae648111efa6dd18c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_302000_73ae0d40648111ef8f6218c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_302000_73771ec0648111efb8f818c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_302000_7342c84f648111ef96ee18c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301964_9fd1b34f647b11efa6d718c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301964_9f4ea09e647b11ef9c4d18c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301964_9ee0eab0647b11ef92b318c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301964_9e7816c0647b11ef8ece18c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301991_805dd1a1647811ef92d918c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301991_7fc53b21647811ef8add18c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301988_fc00bcc0647611efbca218c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301988_fbd12140647611efbfdb18c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301988_fba3cfae647611efa45118c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301988_fb74342e647611ef8ade18c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301988_fb46e29e647611efbb0718c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301976_a2cf03e1647211ef9e8d18c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301976_a2983c70647211efaa3618c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301976_3635c3f0647111ef864b18c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_301964_153f48a1646811efa11418c04d251e75.mp3`: speaker `male_female_mixed`, music `bgm` -> actual `single_male/no_music`
- `page_006_9fd7f921f73911eb8b1cb89a2af90636.mp3`: speaker `multi_speaker_other`, music `bgm` -> actual `single_male/no_music`

## Song / chant mismatches

- actual `song` count: `6`
- `page_301974_1b1c3961685811ef987e18c04d251e75.mp3` predicted `bgm` -> actual `song`
- `page_301988_8911ca80652311ef81c118c04d251e75.mp3` predicted `bgm` -> actual `song`
- `page_301976_7f6a5a8f652011ef8eae18c04d251e75.mp3` predicted `bgm` -> actual `song`
- `000029_Unit 3_Fun Time.mp3` predicted `bgm` -> actual `song`
- `000063_Unit 5_Fun Time.mp3` predicted `bgm` -> actual `song`
- `page_305407_2c3b3a70676d11efabd118c04d251e75.mp3` predicted `bgm` -> actual `song`

## Human-labeled bucket coverage

- `43` x `('en', 'single_female', 'adult', 'no_music')`
- `38` x `('zh', 'single_female', 'adult', 'no_music')`
- `36` x `('en', 'single_male', 'adult', 'no_music')`
- `6` x `('zh', 'single_male', 'adult', 'no_music')`
- `3` x `('en', 'male_female_mixed', 'adult', 'song')`
- `3` x `('en', 'male_female_mixed', 'adult', 'no_music')`
- `3` x `('en', 'single_female', 'child', 'no_music')`
- `2` x `('en', 'single_male', 'child', 'no_music')`
- `2` x `('zh', 'male_female_mixed', 'adult', 'transient_sfx')`
- `2` x `('zh', 'male_female_mixed', 'adult', 'no_music')`
- `1` x `('zh', 'multi_speaker_other', 'unknown', 'transient_sfx')`
- `1` x `('en', 'male_female_mixed', 'unknown', 'song')`
- `1` x `('en', 'male_female_mixed', 'unknown', 'transient_sfx')`
- `1` x `('zh', 'single_female', 'adult', 'transient_sfx')`
- `1` x `('zh', 'single_male', 'adult', 'transient_sfx')`
- `1` x `('mixed', 'multi_speaker_other', 'adult', 'song')`
- `1` x `('en', 'multi_speaker_other', 'child', 'song')`
- `1` x `('en', 'male_female_mixed', 'adult', 'transient_sfx')`

## Missing MECE buckets

- `('zh', 'male_female_mixed', 'no_music')` current `2` / target `6` / missing `4`
- `('zh', 'single_male', 'bgm')` current `0` / target `4` / missing `4`
- `('zh', 'single_female', 'bgm')` current `0` / target `4` / missing `4`
- `('zh', 'male_female_mixed', 'bgm')` current `0` / target `4` / missing `4`
- `('en', 'single_male', 'transient_sfx')` current `0` / target `4` / missing `4`
- `('en', 'single_female', 'transient_sfx')` current `0` / target `4` / missing `4`
- `('en', 'single_male', 'bgm')` current `0` / target `4` / missing `4`
- `('en', 'single_female', 'bgm')` current `0` / target `4` / missing `4`
- `('en', 'male_female_mixed', 'bgm')` current `0` / target `4` / missing `4`
- `('en', 'single_male', 'song')` current `0` / target `4` / missing `4`
- `('en', 'single_female', 'song')` current `0` / target `4` / missing `4`
- `('en', 'male_female_mixed', 'no_music')` current `3` / target `6` / missing `3`
- `('zh', 'multi_speaker_other', 'transient_sfx')` current `1` / target `4` / missing `3`
- `('zh', 'single_male', 'song')` current `0` / target `3` / missing `3`
- `('zh', 'single_female', 'song')` current `0` / target `3` / missing `3`
- `('zh', 'single_male', 'no_music')` current `6` / target `8` / missing `2`
- `('en', 'male_female_mixed', 'transient_sfx')` current `2` / target `4` / missing `2`
- `('zh', 'single_female', 'no_music')` current `38` / target `8` / missing `0`
- `('en', 'single_male', 'no_music')` current `38` / target `8` / missing `0`
- `('en', 'single_female', 'no_music')` current `46` / target `8` / missing `0`
