import main
from commercial_task_v4 import register_commercial
from smart_audio_task import register_smart_audio
from mascot_task import register_mascot
from surgical_task import register_surgical

register_commercial(main.app)
register_smart_audio(main.app)
register_mascot(main.app)
register_surgical(main.app)

if __name__ == "__main__":
    main.app.start()
