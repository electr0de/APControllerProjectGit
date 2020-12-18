from simglucose.simulation.user_interface import simulate
import logging

"""
class PrintStreamHandler(logging.StreamHandler):
    def __init__(self):
        logging.StreamHandler.__init__(self)

    def emit(self, record):
        msg = self.format(record)
        print(msg)



loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
handler = PrintStreamHandler()
log_format = "%(name)s - %(levelname)s - %(message)s"
handler.setFormatter(logging.Formatter(fmt=log_format))

for logger in loggers:
    logger.addHandler(handler)
"""

simulate()
