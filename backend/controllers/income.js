const IncomeSchema = require("../models/IncomeModel")


exports.addIncome = async (req, res) => {
    console.log(req.body);
    const {title, amount, category, description, date} = req.body

    const income = IncomeSchema({
        title,
        amount,
        category,
        description,
        date
    });

    try {
        if(!title || !category || !description || !date) {
            return res.status(400).json({message: "All fields are required!"})
        }
        if(amount <= 0 || !amount === "number") {
            return res.status(400).json({message: "Amount must be a positive number!"})
        }
        //await income.SaveChangesAsync();
        await income.save()
        res.status(200).json({message: "Income Added!"})
    } catch (error) {
        /*res.status(500).json({message: "Server Error!"})*/
        console.log(error);
        res.status(500).json({message: error})
    }

    console.log(income);
}


exports.getIncomes = async (req, res) => {
    try {
        const incomes = await IncomeSchema.find().sort({ date: 1 });
        res.status(200).json(incomes)
    } catch (error) {
        res.status(500).json({message: "Server Error!"})
    }
}

exports.deleteIncome = async (req, res) => {
    const {id} = req.params;
    console.log(req.params)
    IncomeSchema.findByIdAndDelete(id)
        .then((income) => {
            res.status(200).json({message: "Income deleted!"})
        })
        .catch((error) => {
            res.status(500).json({message: "Server Error!"})
        })
}