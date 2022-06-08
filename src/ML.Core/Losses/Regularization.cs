namespace ML.Core.Losses
{
    /// <summary>
    ///     权重约束
    /// </summary>
    public enum Regularization
    {
        None = 0,
        Lasso_L1 = 1,
        Ridge_L2 = 2,
        ElasticNet_LP = 3
    }
}