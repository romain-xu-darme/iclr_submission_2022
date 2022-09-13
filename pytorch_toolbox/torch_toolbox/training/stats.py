from typing import List, Any, Optional


class ProgressMeter(object):
    """ Progress meter

    :param num_batches: Number of expected batches
    :param meters: List of metrics
    :param prefix: Progress bar metric
    :param linebreak: Add line break after display
    """

    def __init__(self,
                 num_batches: int,
                 meters: List[Any],
                 prefix: Optional[str] = "",
                 linebreak: Optional[bool] = False,
                 ) -> None:
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.linebreak = linebreak
        entries = [self.prefix, 'Batch'] + [meter.name for meter in self.meters]
        print(' '.join(entries))

    def display(self, batch: int, last: Optional[bool] = False) -> None:
        """ Update progress

        :param batch: Batch index
        :param last: Last update
        """
        entries = ['\t' + self.batch_fmtstr.format(batch + 1)]
        entries += [str(meter) for meter in self.meters]
        end = '\n' if self.linebreak or last else '\r'
        print(' '.join(entries), end=end)

    @staticmethod
    def _get_batch_fmtstr(num_batches: int) -> str:
        """ Get format string

        :param num_batches (int): Number of expected batches
        :return: Format string
        """
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class AverageMeter(object):
    """ Computes and stores the average and current value

    :param name: Meter name
    :param fmt: Display format

    """

    def __init__(self, name: str, fmt: str = ':f', avg_only: bool = True) -> None:
        self.name = name
        self.fmt = fmt
        self.avg_only = avg_only
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self) -> None:
        """ Reset meter """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: Any, n: int = 1) -> None:
        """ Update meter

        :param val: Meter value for batch
        :param n: Batch size
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        """ Return string of meter average value """
        if self.avg_only:
            fmtstr = '{avg' + self.fmt + '}'
        else:
            fmtstr = '{val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
