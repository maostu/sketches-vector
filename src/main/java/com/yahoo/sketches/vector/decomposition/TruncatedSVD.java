package com.yahoo.sketches.vector.decomposition;

import org.ojalgo.matrix.decomposition.QR;
import org.ojalgo.matrix.decomposition.SingularValue;
import org.ojalgo.matrix.store.MatrixStore;
import org.ojalgo.matrix.store.PrimitiveDenseStore;
import org.ojalgo.random.Normal;

final class TruncatedSVD {
  private static final int DEFAULT_NUM_ITER = 8;

  private long nIter_;

  //private SingularValue<Double> svd;
  private double[] sv_;
  private MatrixStore<Double> Vt_;

  private TruncatedSVD(final long nIter) {
    if (nIter < 1) {
      throw new IllegalArgumentException("nIter must be a positive integer, found: " + nIter);
    }
    nIter_ = nIter;
  }

  public static TruncatedSVD make(final MatrixStore<Double> A) {
    return make(A, DEFAULT_NUM_ITER);
    //return make(A, Math.min(A.countColumns(), A.countRows()) / 2);
  }

  public static TruncatedSVD make(final MatrixStore<Double> A, final long numIter) {
    return new TruncatedSVD(numIter);
  }

  public void compute(final MatrixStore<Double> A, final int k) {
    if (k < 1) {
      throw new IllegalArgumentException("k must be a positive integer, found: " + k);
    }

    // want to iterate on smaller dimension of A (n x d)
    // currently, error in constructor if d < n, so n is always the smaller dimension
    final long d = A.countColumns();
    final long n = A.countRows();
    final PrimitiveDenseStore block = PrimitiveDenseStore.FACTORY.makeFilled(d, k, new Normal(0.0, 1.0));

    // orthogonalize for numeric stability
    final QR<Double> qr = QR.PRIMITIVE.make(block);
    qr.decompose(block);
    qr.getQ().supplyTo(block);

    final PrimitiveDenseStore T = PrimitiveDenseStore.FACTORY.makeZero(n, k);

    for (int i = 0; i < nIter_; ++i) {
      A.multiply(block).supplyTo(T);
      A.transpose().multiply(T).supplyTo(block);

      // again, just for stability
      qr.decompose(block);
      qr.getQ().supplyTo(block);
    }

    // Rayleigh-Ritz postprocessing
    A.multiply(block).supplyTo(T);

    final SingularValue<Double> svd = SingularValue.make(T);
    svd.compute(T);

    sv_ = new double[k];
    svd.getSingularValues(sv_);

    //block.multiply(svd.getQ2().transpose()).supplyTo(block);
    Vt_ = block.multiply(svd.getQ2()).transpose();
  }

  public MatrixStore<Double> getVt() {
    //return svd.getQ2().transpose();
    return Vt_;
  }

  public void getSingularValues(final double[] values) {
    //svd.getSingularValues(values);
    System.arraycopy(sv_, 0, values, 0, sv_.length);
  }
}
