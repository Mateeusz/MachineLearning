class Instance:
    id: int
    featureValues: list
    cancerType: int

    def __init__(self, id, featureValues, cancerType, defaultValueOfInvalidFeature):
        self.id = id
        self.featureValues = featureValues
        self.cancerType = cancerType

        try:
            index = self.featureValues.index('?')
            self.featureValues[index] = defaultValueOfInvalidFeature
        except ValueError as ve:
            pass
        try:
            for x in range(self.featureValues.__len__()):
                if float(self.featureValues[x]) < 1.0:
                    print(self.featureValues[x])
                    self.featureValues[x] = defaultValueOfInvalidFeature
                    print('AFTER', self.featureValues[x])

        except ValueError as ve:
            pass

    def __str__(self):
        print(self.id, self.featureValues, self.cancerType)

    def getId(self):
        return self.id

    def getFeatureValues(self):
        return self.featureValues

    def getFeature(self, feature):
        value = self.getFeatureValues()[feature]

        return value if value != '?' else 1

    def getCancerType(self):
        return self.cancerType

