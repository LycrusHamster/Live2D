import { DocumentBuilder, SwaggerModule } from '@nestjs/swagger';

export default (app: any, path = `/api/v1/doc`): void => {
  const document = SwaggerModule.createDocument(
    app,
    new DocumentBuilder()
      .setTitle('Capy API')
      .setDescription('Capy API')
      .setVersion('1.0')
      .addServer(`/api/v1`)
      .build(),
    { ignoreGlobalPrefix: true, deepScanRoutes: true },
  );

  //fs.writeFileSync('./swagger-spec.json', JSON.stringify(document, null, 2));

  SwaggerModule.setup(path, app, document);
};
