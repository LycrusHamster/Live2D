import { NestFactory } from '@nestjs/core';
import { ValidationPipe } from '@nestjs/common';
import { NestExpressApplication } from '@nestjs/platform-express';
import * as express from 'express';
import helmet from 'helmet';
import * as bodyParser from 'body-parser';

import { AppModule } from './app.module';
import initSwagger from './swagger';

declare const module: any;

async function bootstrap() {
  const app = await NestFactory.create<NestExpressApplication>(AppModule);

  app.use(bodyParser.json({ limit: '50mb' }));
  app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));

  app.use(express.json());

  app.use(express.urlencoded({ extended: true }));
  app.enableCors();
  app.use(
    helmet({
      contentSecurityPolicy: false,
    }),
  );
  app.setGlobalPrefix('/api/v1');

  initSwagger(app);

  if (module.hot) {
    module.hot.accept();
    module.hot.dispose(() => app.close());
  }

  await app.listen(9000, '0.0.0.0');

  console.log(
    `listen http://localhost:9000/api/v1`,
  );
}

bootstrap();
