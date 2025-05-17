import {Module} from '@nestjs/common';
import {ConfigModule} from '@nestjs/config';
import * as process from 'node:process';
import {HuggingFaceModule} from "./modules/huggingFace/huggingFace.module";

@Module({
    imports: [
        ConfigModule.forRoot({
            envFilePath: [
                // `.env.${process.env.NODE_ENV}`,
                '.env',
            ],
            isGlobal: true,
        }),
        HuggingFaceModule
    ],
    providers: [],
})

export class AppModule {
}
