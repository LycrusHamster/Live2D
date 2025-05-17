import { ApiProperty } from '@nestjs/swagger';

export class AskDto {
  @ApiProperty({
    description: '语音的Float32Array转Buffer转Base64',
    example: false,
  })
  audio: string;

}
