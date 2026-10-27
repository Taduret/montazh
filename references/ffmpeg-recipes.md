# Рецепты ffmpeg для монтажа

Готовые команды на случай, когда нужно поправить руками то, что скилл не делает.
Все примеры — для ffmpeg 6+; в Windows пути в кавычках.

## Диагностика

```bash
# всё, что нужно знать о файле
ffprobe -v error -print_format json -show_format -show_streams "input.mp4"

# длительность без разбора JSON
ffprobe -v error -show_entries format=duration -of csv=p=0 "input.mp4"

# есть ли аудио
ffprobe -v error -select_streams a -show_entries stream=codec_name -of csv=p=0 "input.mp4"
```

## Нарезка

```bash
# вырезать фрагмент (быстро, перекодируя)
ffmpeg -ss 12.5 -t 4.2 -i input.mp4 -c:v libx264 -crf 17 -preset veryfast -an out.mp4

# вырезать без перекодирования (только по ключевым кадрам — режет «примерно»)
ffmpeg -ss 12.5 -t 4.2 -i input.mp4 -c copy out.mp4

# склейка одинаковых по параметрам кусков
printf "file '%s'\n" seg_*.mp4 > list.txt
ffmpeg -f concat -safe 0 -i list.txt -c copy out.mp4
```

## Кадр под платформу

```bash
# вертикаль 9:16 из квадрата (обрезаем по центру)
ffmpeg -i in.mp4 -vf "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" -c:v libx264 -crf 18 out.mp4

# 16:9 из вертикали с размытым фоном (без обрезки)
ffmpeg -i in.mp4 -filter_complex \
 "[0:v]scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,gblur=sigma=30[bg];\
  [0:v]scale=-2:1080[fg];[bg][fg]overlay=(W-w)/2:(H-h)/2" out.mp4
```

## Ритм и скорость

```bash
# замедление в 1.25 раза, звук в тон
ffmpeg -i in.mp4 -vf "setpts=1.25*PTS" -af "atempo=0.8" out.mp4

# ускорение в 1.5 раза
ffmpeg -i in.mp4 -vf "setpts=PTS/1.5" -af "atempo=1.5" out.mp4

# плавное изменение скорости от 1.0 к 2.0 (без звука)
ffmpeg -i in.mp4 -vf "setpts='PTS/(1+1*t/2)'" -an out.mp4
```

## Зум и движение камеры

```bash
# медленный наезд на статичный кадр (Ken Burns), 5 c
ffmpeg -loop 1 -i photo.jpg -vf \
 "zoompan=z='min(1+0.0006*on,1.12)':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=150:s=1080x1080:fps=30" \
 -t 5 -c:v libx264 -crf 18 out.mp4

# наезд на видео: каждый входной кадр → один выходной
ffmpeg -i in.mp4 -vf \
 "zoompan=z='1+0.05*on/60':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s=1080x1080:fps=30" out.mp4
```

## Переходы

```bash
# один наплыв между двумя файлами, 0.4 c
ffmpeg -i a.mp4 -i b.mp4 -filter_complex \
 "[0:v][1:v]xfade=transition=fade:duration=0.4:offset=2.6" -c:v libx264 out.mp4

# доступные переходы: fade, wipeleft, slideup, circleopen, dissolve, smoothleft …
ffmpeg -h filter=xfade
```

## Звук

```bash
# нормализация громкости в два прохода (точнее, чем один)
ffmpeg -i music.mp3 -af loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json -f null -
# из вывода берём measured_* и подставляем:
ffmpeg -i music.mp3 -af loudnorm=I=-14:TP=-1.5:LRA=11:measured_I=-9.2:measured_TP=-0.8:\
measured_LRA=6.1:measured_thresh=-20.1:offset=0.3 -ar 48000 out.wav

# затухание в конце 2 c
ffmpeg -i music.mp3 -af "afade=t=out:st=58:d=2" out.mp3

# подложить музыку под видео, обрезав по видео
ffmpeg -i video.mp4 -i music.mp3 -map 0:v -map 1:a -c:v copy -c:a aac -b:a 192k -shortest out.mp4
```

## Субтитры и текст

```bash
# вжечь внешние субтитры
ffmpeg -i in.mp4 -vf "subtitles=subs.srt:force_style='FontName=Inter,FontSize=22'" out.mp4

# убрать захардкоженные субтитры обрезкой нижней полосы (если они внизу)
ffmpeg -i in.mp4 -vf "crop=iw:ih*0.86:0:0,scale=1080:-2" out.mp4
```

## Проверка результата

```bash
# сколько кадров и какая длительность
ffprobe -v error -count_frames -select_streams v:0 \
 -show_entries stream=nb_read_frames,duration,width,height -of default=noprint_wrappers=1 out.mp4

# сетка кадров для быстрого просмотра
ffmpeg -i out.mp4 -vf "fps=1/4,scale=240:-2,tile=6x4" -frames:v 1 sheet.jpg

# сплошной чёрный кадр искать так
ffmpeg -i out.mp4 -vf "blackdetect=d=0.1:pix_th=0.1" -f null - 2>&1 | grep blackdetect
```
