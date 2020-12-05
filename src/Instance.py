class Instance:
    id: int
    featureValues: list
    cancerType: int
    cancerClass: str

    def __init__(self, id, featureValues, cancerType, defaultValueOfInvalidFeature):
        self.id = id
        self.featureValues = featureValues
        self.cancerType = cancerType

        if int(cancerType) == 2:
            self.cancerClass = 'B'
        else:
            self.cancerClass = 'M'

        try:
            index = self.featureValues.index('?')
            self.featureValues[index] = defaultValueOfInvalidFeature
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

    def getCancerClass(self):
        return self.cancerClass