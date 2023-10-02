namespace MLRestaurant
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Trainer trainer = new Trainer();
            Predictor predictor = new Predictor();

            trainer.Train("C:\\Users\\mort286f\\source\\repos\\MLRestaurant\\MLRestaurant\\Data\\sampledata.csv");

            var input = Console.ReadLine();

            predictor.Predict(input);


        }
    }
}