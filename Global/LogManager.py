#!/usr/bin/env python
import logging
import logging.handlers
import os
import sys
class Log():
    # FILE_PATH = os.path.expanduser('~') + "/PythonLog/"
    FILE_PATH = os.path.dirname(__file__) + "/Log/"
    if not (os.path.isdir(FILE_PATH)):
        os.makedirs(os.path.join(FILE_PATH))
        pass
    
    log_max_size = 1000 * 512
    log_file_count = 50

    # region Error Log
    log_error = logging.getLogger('log_error')
    log_error.setLevel(logging.ERROR)
    formatter_error = logging.Formatter('[%(asctime)s][%(levelname)s] (%(filename)s:%(lineno)d) >>> %(message)s')

    streamHandler_error = logging.StreamHandler()
    streamHandler_error.setFormatter(formatter_error)

    LogHandler_error = logging.handlers.RotatingFileHandler(
        filename=FILE_PATH +'/error.txt',
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode="w",
        )

    LogHandler_error.setFormatter(formatter_error)
    LogHandler_error.suffix = "%Y%m%d"

    log_error.addHandler(LogHandler_error)
    log_error.addHandler(streamHandler_error)
    #endregion

    # region System Log
    log_system = logging.getLogger('log_system')
    log_system.setLevel(logging.DEBUG)
    formatter_system = logging.Formatter('[%(asctime)s][%(levelname)s] (%(filename)s:%(lineno)d) >>> %(message)s')

    streamHandler_system = logging.StreamHandler()
    streamHandler_system.setFormatter(formatter_system)

    LogHandler_system = logging.handlers.RotatingFileHandler(
        filename=FILE_PATH +'/history.txt',
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode="w",
        )
    LogHandler_system.setFormatter(formatter_system)
    LogHandler_system.suffix = "%Y%m%d"

    log_system.addHandler(LogHandler_system)
    log_system.addHandler(streamHandler_system)
    #endregion

    # region Topic Log
    log_topic = logging.getLogger('log_topic')
    log_topic.setLevel(logging.DEBUG)
    formatter_topic = logging.Formatter('[%(asctime)s] >>> %(message)s')

    streamHandler_topic = logging.StreamHandler()
    streamHandler_topic.setFormatter(formatter_topic)

    LogHandler_topic = logging.handlers.RotatingFileHandler(
        filename=FILE_PATH + '/topic.txt',
        maxBytes=log_max_size,
        backupCount=log_file_count,
        mode="w",
    )
    LogHandler_topic.setFormatter(formatter_topic)
    LogHandler_topic.suffix = "%Y%m%d"

    log_topic.addHandler(LogHandler_topic)
    log_topic.addHandler(streamHandler_topic)
    # endregion