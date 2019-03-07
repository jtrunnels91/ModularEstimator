import numpy as np

class covarianceContainer():
    _recougnizedForms=['covariance', 'cholesky']
    def __init__(self, covMat, form):
        self.value = covMat
        if form not in covarianceContainer._recougnizedForms:
            raise ValueError('Unrecougnized covariance form %s' %form)
        self.form = form
        return
    
    def convertCovariance(self, newForm):
        if self.form == newForm:
            newCov = self
        elif newForm == 'covariance':
            if self.form == 'cholesky':
                newCov = covarianceContainer(
                    self.value.dot(self.value.transpose()),
                    form=newForm
                )
            else:
                raise ValueError('Unrecougnized covariance form %s' %self.form)

        elif newForm == 'cholesky':
            if self.form == 'covariance':
                try:
                    if not np.any(self.value):
                        newCov = covarianceContainer(
                            self.value,
                            form=newForm
                    )
                    else:
                        newCov = covarianceContainer(
                            np.linalg.cholesky(self.value),
                            form=newForm
                        )
                except:
                    print(self)
                    print("error converting covariance!")
                    raise ValueError("error converting covariances")
            else:
                raise ValueError('Unrecougnized covariance form %s' %self.form)
        return newCov

    def __add__(self, other):
        if other.form != self.form:
            other = other.convertCovariance(self.form)
        
        if self.form == 'covariance':
            mySum = covarianceContainer(self.value + other.value, 'covariance')
        else:
            matStack = np.vstack([self.value, other.value])
            QR = np.linalg.qr(matStack)
            mySumValue = QR[1].transpose()
            if mySum[0,0] < 0:
                mySumValue = - mySumValue
            mySum = covarianceContainer(mySumValue, 'cholesky')
            
        return mySum
    
    def __getitem__(self, key):
        subMat = self.value[key]
        return covarianceContainer(subMat, self.form)
    def __setitem__(self, key, newVal):
        if isinstance(newVal, covarianceContainer):
            self.value[key] = newVal.convertCovariance(self.form).value
        else:
            self.value[key] = newVal

    def __repr__(self):
        repString = 'covarianceContainer (form=%s, value=\n' %self.form
        repString += '%s)' %self.value
        return repString

    def mahalanobisDistance(self, dX):
        if self.form == 'covariance':
            MSquared = dX.transpose().dot(
                np.linalg.inv(self.value)
            ).dot(dX)
        elif self.form == 'cholesky':
            V = dX.transpose().dot(np.linalg.inv(self.value))
            MSquared = V.dot(V)
        M = np.sqrt(MSquared)
        return(M)

    @property
    def shape(self):
        return self.value.shape
