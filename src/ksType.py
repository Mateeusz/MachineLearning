class ksType:
    statistic: int
    pValue: int
    id: int

    def __init__(self, paramID, statistic, pvalue):
        self.statistic = statistic
        self.pValue = pvalue
        self.id = paramID

    def getStatistic(self):
        return self.statistic

    def getPValue(self):
        return self.pValue

    def getParamID(self):
        return self.id