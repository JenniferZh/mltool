import numpy as np
import math
from sklearn.cluster import KMeans

class GMM(object):

    def __init__(self, dim = None, ncomps = None, data = None, method = None, params = None):
        if not params is None:
            self.comps = params['comps']
            self.ncomps = params['ncomps']
            self.dim = params['dim']

        elif not data is None:
            self.data = data
            self.dim = dim
            self.ncomps = ncomps
            self.comps = []

            if method is "kmeans":
                #initialize each component using kmeans
                kmeans = KMeans(n_clusters=ncomps, random_state=0).fit(data)

                #initial_mean = kmeans.cluster_centers_
                labels = kmeans.labels_
                n = data.shape[0]

                for i in range(self.ncomps):
                    indices, = np.where(labels==i)

                    covariance = np.cov(data[indices].T)
                    prior = indices.size / n

                    compdict = {}
                    compdict['prior'] = prior
                    compdict['mean'] = sum(data[indices], 0)/indices.size
                    compdict['cov'] = covariance

                    self.comps.append(compdict)

    #compute the value of x in ith component
    def pdf(self, x, ith):
        compdict = self.comps[ith]
        mean = compdict['mean']
        cov = compdict['cov']
        factor = 1/math.sqrt(2*np.pi)*math.sqrt(np.linalg.det(cov))

        return factor*math.exp(-0.5*np.dot(np.matmul(x-mean, np.linalg.inv(cov)), (x-mean)))

    def maxlikelihood(self):
        sum = 0
        for i in range(self.data.shape[0]):
            for k in range(self.ncomps):
                sum += self.comps[k]['prior']*self.pdf(self.data[i], k)
        return sum

    def em(self):

        n = self.data.shape[0]
        d = self.dim
        k = self.ncomps
        steps = 100

        for st in range(steps):
            # e step
            gamma = np.zeros(shape=(n, self.dim))
            for i in range(n):
                sum = 0
                for j in range(k):

                    sum += self.comps[j]['prior']*self.pdf(self.data[i], j)
                for j in range(k):
                    gamma[i, j] = self.comps[j]['prior']*self.pdf(self.data[i], j)/sum


            #m step
            nk = np.sum(gamma, 0)


            for t in range(k):
                tmp = gamma[:, t]
                tmp = np.tile(tmp, (d, 1))



                self.comps[t]['mean'] = np.sum(self.data*tmp.T, 0)/nk[t]

                tmp2 = np.tile(self.comps[t]['mean'], (n, 1))
                self.comps[t]['cov'] = np.dot(tmp*(self.data.T-tmp2.T), (self.data-tmp2))/nk[t]

                self.comps[t]['prior'] = nk[t]/np.sum(nk)

            print(self.maxlikelihood())






def main():
    dataset = np.array([[4,4],[2,1],[5,4],[2,0],[4,5],[1,1]])

    gmm = GMM(dim=2, ncomps=2, data=dataset, method="kmeans")
    gmm.em()

    print(gmm.comps)

    #print(gmm.pdf([1,1],1))




main()