# main.py
from gen_plan import gen_plan
from gen_KG_graphRAG import build_KG
from use_KG import use_KG

from saver import saver
from config import FLAGS
from utils import OurTimer, get_root_path, report_save_dir
import traceback
import sys
sys.path.insert(0, '../')

timer = OurTimer()

def main():
    if FLAGS.task == 'gen_plan':
        gen_plan()
    elif FLAGS.task == 'build_KG':
        build_KG()
    elif FLAGS.task == 'use_KG':
        use_KG()
    else:
        raise NotImplementedError()

if __name__ == "__main__":
    main()
