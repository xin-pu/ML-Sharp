namespace ML.Core.Data.Loader
{
    public abstract class TextLoader
    {
    }


    public interface IDataLoad<in T>
    {
        DataSet Load(T input);
    }
}