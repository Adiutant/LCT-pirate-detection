## Инструкция

Первый запуск

```bash
# Важно, укажите зависимости в файле next.config.mjs в env
# Установка зависимостей
npm i
```

Зависимости (next.config.mjs)
```js
// Пример
const nextConfig = {
    env: {
        HTTP_VIDEO_DOWNLOAD: 'http://caddy:5291/api/videos/download', // адрес пополнения БД без модерации
        WS_VIDEO_DOWNLOAD: 'http://caddy:5291/api/status/currentStatus', // адрес проверки статуса пополнения
        HTTP_VIDEO_MODERATION: 'http://caddy:5291/api/videos/verify', // адрес пополнения БД с модерацией
        WS_VIDEO_MODERATION: 'http://caddy:5291/api/status/currentStatus', // адрес проверки статуса модерации
    }
};
```

```bash
# Запуск в dev режиме
npm run dev
```

```bash
# Сборка приложения
npm run build
# Запуск сборки
npm run start
```

По умолчанию сайт должен открыть на [http://localhost:3000](http://localhost:3000), однако в случае, если порт занят, то он моет занять порт на 1 выше (пример: если порт 3000 занят, то он займёт 3001).
