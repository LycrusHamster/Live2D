import {Body, Controller, Get, Param, Post, Query, Req} from '@nestjs/common';
import {HuggingFaceService} from './huggingFace.service';
import { Request } from 'express';
import {AskDto} from "./dto/ask.dto";

@Controller('huggingFace')
export class HuggingFaceController {
    constructor(
        private readonly huggingFaceService: HuggingFaceService,
    ) {
    }

    @Post('ask')
    async ask(
        @Body() req: AskDto
    ) {

        console.log(`incoming ask`)

        const buffer = Buffer.from(req.audio,`base64`)

        // 转换为 Float32Array（注意单位：4 字节）
        const floatArray = new Float32Array(
            buffer.buffer,
            buffer.byteOffset,
            buffer.length / Float32Array.BYTES_PER_ELEMENT
        );

        // const data = [...floatArray];

        console.log('floatArray dimension: ', floatArray.length);

        const ret = await this.huggingFaceService.handle(floatArray)

        let audioDataBase64 = Buffer.from(ret.audioData).toString('base64')

        return {
            asr: ret.asr,
            answer: ret.answer,
            audio: audioDataBase64
        }
    }

}
