export class AudioRecorderUtil {
  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private analyserNode: AnalyserNode | null = null;
  private audioContext: AudioContext | null = null;
  private sourceNode: MediaStreamAudioSourceNode | null = null;
  private stream: MediaStream | null = null;

  async startRecording(): Promise<AnalyserNode> {
    this.audioChunks = [];

    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        sampleRate: 44100,
      },
    });

    this.audioContext = new AudioContext({ sampleRate: 44100 });
    this.sourceNode = this.audioContext.createMediaStreamSource(this.stream);
    this.analyserNode = this.audioContext.createAnalyser();
    this.analyserNode.fftSize = 256;
    this.sourceNode.connect(this.analyserNode);

    this.mediaRecorder = new MediaRecorder(this.stream);
    this.mediaRecorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        this.audioChunks.push(event.data);
      }
    };

    this.mediaRecorder.start(100);
    return this.analyserNode;
  }

  async stopRecording(): Promise<{ audioData: Float32Array; sampleRate: number }> {
    return new Promise((resolve, reject) => {
      if (!this.mediaRecorder || !this.audioContext) {
        reject(new Error("No recording in progress"));
        return;
      }

      const ctx = this.audioContext;

      this.mediaRecorder.onstop = async () => {
        try {
          const audioBlob = new Blob(this.audioChunks, { type: "audio/webm" });
          const arrayBuffer = await audioBlob.arrayBuffer();
          const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
          const audioData = audioBuffer.getChannelData(0);

          resolve({
            audioData,
            sampleRate: audioBuffer.sampleRate,
          });
        } catch (err) {
          reject(err);
        }
      };

      this.mediaRecorder.stop();
      this.stream?.getTracks().forEach((track) => track.stop());
    });
  }

  getAnalyserNode(): AnalyserNode | null {
    return this.analyserNode;
  }

  cleanup() {
    this.stream?.getTracks().forEach((track) => track.stop());
    this.audioContext?.close();
    this.mediaRecorder = null;
    this.analyserNode = null;
    this.audioContext = null;
    this.sourceNode = null;
    this.stream = null;
  }
}
