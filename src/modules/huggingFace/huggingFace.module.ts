import {Module} from '@nestjs/common';
import {HuggingFaceService} from './huggingFace.service';
import {HuggingFaceController} from './huggingFace.controller';
import {ConfigModule, ConfigService} from '@nestjs/config';

@Module({
    imports: [ConfigModule],
    controllers: [HuggingFaceController],
    providers: [HuggingFaceService],
})
export class HuggingFaceModule {
}
