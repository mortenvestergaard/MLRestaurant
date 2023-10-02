using Microsoft.ML;
using Microsoft.VisualBasic;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MLRestaurant
{
    public class BaseML
    {
        protected static string ModelPath => Path.Combine(AppContext.BaseDirectory, "model.mdl");
        protected readonly MLContext MlContext;
        protected BaseML()
        {
            MlContext = new MLContext(20);
        }
    }

}
