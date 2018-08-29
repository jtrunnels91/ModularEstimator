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
                newCov = covarianceContainer(
                    np.linalg.cholesky(self.value),
                    form=newForm
                )
            else:
                raise ValueError('Unrecougnized covariance form %s' %self.form)
        return newCov

    def __getitem__(self, key):
        subMat = self.value[key]
        return covarianceContainer(subMat, self.form)
