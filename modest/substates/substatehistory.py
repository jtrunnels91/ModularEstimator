class SubStateHistory():

    def __init__(
            self,
            data=None
            ):
        self.typecheck(data)

        self.columnDict = {}

        for key in data:
            if isinstance(data[key], list):
                self.columnDict[key] = data[key]
            else:
                self.columnDict[key] = [data[key]]

        return

    def __getitem__(self, key):

        returnDict = {}
        if isinstance(key, str):
            returnDict = self.columnDict[key]
        else:
            for col in self.columnDict:
                returnDict[col] = self.columnDict[col][key]
        return returnDict

    def __setitem__(self, key, item):
        self.typecheck(item)
        for col in item:
            self.columnDict[col][key] = item[col]

        return

    def append(self, item):
        self.typecheck(item)
        for col in item:
            if isinstance(item[col], list):
                self.columnDict[col].extend(item[col])
            else:
                self.columnDict[col].append(item[col])
        return

    def typecheck(self, item):
        if isinstance(item, dict):
            pass
        else:
            raise TypeError(
                "Received unknown data type %s" % type(item)
                )
        
        return
    
