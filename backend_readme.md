# Backend сервис для хакатона Лидеры цифровой трансформации

### Описание
Сервис принимает видео от пользователя и сохраняет его в s3 сервис, после чего передает его на анализ для ml. 
После анализа от ml собирает данные и передает в frontend. 

### Стек технологий
Asp net core, Quartz, Postgresql, S3, Caddy

### Развертывание
Сервис доступен из публичного репозитория docker.io `eiparfenov/lct-backend`

Для развертывания рекомендуется использовать docker-compose.yml файл из проекта, который находится в папке services/lct_service.

### Конфигурация
Конфигурация через переменные окружения
* `ConnectionStrings__PostgresDb`: строка подключения к базе
* `S3Options__AccessKeyId`: ид ключа для s3
* `S3Options__SecretAccessKey`: ключ к s3
* `S3Options__ServiceUrl`: url s3 сервиса
* `S3Options__BucketName`: имя бакета
* `UrlGeneratorOptions__Template`: шаблон для генерации ссылки `https://3ce0-109-252-24-49.ngrok-free.app/videos-bucket/videos/{0}.mp4`
