using System.Data;

namespace ML.Core.Data.Loader
{
    /// <summary>
    ///     A text loader for get IEnum<T> from txt file.
    /// </summary>
    public class TextLoader
    {
    }


    public interface IDataLoad<in T>
    {
        DataSet Load(T input);
    }
}