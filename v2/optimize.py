import sys, os, logging

from hyperopt import hp, fmin, tpe
from hyperopt import STATUS_OK, STATUS_FAIL

from ghosh import GhoshModel, MODEL_FILE

LEARN = 'a'
KERNEL = 'kernel'
BATCH = 'batch'

log = logging.getLogger('OPTIMIZER')

class Optimizer:

    def __init__(self, model, output_dir, train_path, validation_path):
        self.model = model
        self.output_dir = output_dir
        self.train_path = train_path
        self.validation_path = validation_path

    def objective(self, args):

        learn = args[LEARN]
        kernel = args[KERNEL]
        batch = args[BATCH]

        log.info("Running objective: %s, %s, %s" % (str(batch), str(kernel), str(learn)))
        log.info("  Batch size:     %s" % str(batch))
        log.info("  Kernel size:    %s" % str(kernel))
        log.info("  Learning rate:  %s" % str(learn))

        if learn is None or kernel is None or batch is None:
            return {'status': STATUS_FAIL}

        out_path = self.get_output_dir(learn, kernel, batch)
        log.info("Outputting to %s" % out_path)

        if not os.path.isdir(out_path):
            os.mkdir(out_path)

        log.info("Creating model")
        model = self.model(self.train_path,
                           self.validation_path,
                           learning_rate=learn,
                           kernel_size=kernel,
                           batch_size=batch,
                           output_dir=out_path,
                           retrain=True)

        log.info("Running model")
        f1_score, accuracy = model.run()

        result = {
            'status': STATUS_OK,
            'loss': 1 - (sum(f1_score) / float(len(f1_score))),
            'attachments': {
                'model': model.get_file_path(MODEL_FILE),
                'dir': out_path,
                'f1_score': f1_score,
                'accuracy': accuracy
            }
        }

        log.info(result)
        return result

    def get_output_dir(self, a, b, c):
        return os.path.join(self.output_dir, "optimize_%s_%s_%s" % (a,b,c))

def main():

    #Setup log
    dir_path = os.path.dirname(os.path.realpath(__file__))
    fmt = "%(levelname) -10s %(asctime)s %(module)s:%(lineno)s %(funcName)s %(message)s"
    handler = logging.FileHandler(os.path.join(dir_path, 'optimizer.log'), mode='w')
    handler.setFormatter(logging.Formatter(fmt))
    log.addHandler(handler)
    log.setLevel(logging.DEBUG)

    try:
        optimizer = Optimizer(GhoshModel, sys.argv[3], sys.argv[1], sys.argv[2])
    except Exception as e:
        log.error(e)

    space = {
        LEARN: hp.uniform(LEARN, 0.0000001, 0.0001),
        KERNEL: hp.quniform(KERNEL, 8, 3, 1),
        BATCH: hp.quniform(BATCH, 128, 4, 1)
    }
    log.info("Space:")
    log.info(space)

    best = fmin(optimizer.objective,
         space=space,
         algo=tpe.suggest,
         max_evals=100)

    print(best)
    log.info(str(best))


if __name__ == '__main__':
    main()
