/**
 * BeanHealth CLR — self-exam voice and tones
 * ==========================================
 *
 * Speech (Web Speech API) and short tones (Web Audio) for the voice-guided
 * self exam, where the patient keeps their eyes on the torch and cannot read
 * the screen. Browsers allow audio only after a tap, so unlock() must run in
 * a click handler before anything is spoken.
 */

type AudioContextCtor = typeof AudioContext;

export class ExamVoice {
  private ctx: AudioContext | null = null;
  // Chrome can drop an utterance's end event if nothing references it.
  private pending = new Set<SpeechSynthesisUtterance>();

  /** True while any utterance this voice queued is still being spoken. */
  get speaking(): boolean {
    return this.pending.size > 0;
  }

  /** Start audio from a user gesture. */
  unlock(): void {
    if (!this.ctx) {
      const w = window as Window & { webkitAudioContext?: AudioContextCtor };
      const Ctor = window.AudioContext ?? w.webkitAudioContext;
      if (Ctor) this.ctx = new Ctor();
    }
    void this.ctx?.resume().catch(() => undefined);
    window.speechSynthesis?.cancel();
  }

  /**
   * Speak `text`; resolves when it has been spoken (or was cut off). With
   * `interrupt`, anything still queued is cancelled first.
   */
  say(text: string, interrupt = true): Promise<void> {
    const synth = typeof window !== "undefined" ? window.speechSynthesis : undefined;
    if (!synth) return Promise.resolve();
    if (interrupt) synth.cancel();
    const u = new SpeechSynthesisUtterance(text);
    u.rate = 0.95;
    return new Promise((resolve) => {
      let done = false;
      const finish = () => {
        if (done) return;
        done = true;
        clearTimeout(timer);
        this.pending.delete(u);
        resolve();
      };
      // Some engines never fire the end event; give up after a generous estimate.
      const timer = setTimeout(finish, 2000 + text.length * 110);
      u.onend = finish;
      u.onerror = finish;
      this.pending.add(u);
      synth.speak(u);
    });
  }

  /** A short sine tone. */
  beep(freq: number, ms = 80, volume = 0.2): void {
    const ctx = this.ctx;
    if (!ctx) return;
    const t = ctx.currentTime;
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = "sine";
    osc.frequency.value = freq;
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.exponentialRampToValueAtTime(volume, t + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + ms / 1000);
    osc.connect(gain).connect(ctx.destination);
    osc.start(t);
    osc.stop(t + ms / 1000 + 0.02);
  }

  /** Two rising tones: a position was recorded. */
  chime(): void {
    this.beep(880, 110);
    setTimeout(() => this.beep(1320, 160), 120);
  }

  /** Stop speaking (tones already started finish by themselves). */
  silence(): void {
    window.speechSynthesis?.cancel();
    this.pending.clear();
  }

  close(): void {
    this.silence();
    void this.ctx?.close().catch(() => undefined);
    this.ctx = null;
  }
}
