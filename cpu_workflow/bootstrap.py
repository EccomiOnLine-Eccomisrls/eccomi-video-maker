import main
from commercial_task_v4 import register_commercial
from smart_audio_task import register_smart_audio
from mascot_task import register_mascot

register_commercial(main.app)
register_smart_audio(main.app)
register_mascot(main.app)

if __name__ == "__main__":
    main.app.start()
