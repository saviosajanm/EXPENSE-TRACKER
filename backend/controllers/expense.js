const ExpenseSchema = require("../models/expenseModel")


exports.addExpense = async (req, res) => {
    //console.log(req.body);
    const {title, amount, category, description, date} = req.body

    const expense = ExpenseSchema({
        title,
        amount,
        category,
        description,
        date
    })

    try {
        if(!title || !category || !description || !date) {
            return res.status(400).json({message: "All fields are required!"})
        }
        if(amount <= 0 || !amount === "number") {
            return res.status(400).json({message: "Amount must be a positive number!"})
        }
        await expense.save()
        res.status(200).json({message: "Expense Added!"})
    } catch (error) {
        res.status(500).json({message: "Server Error!"})
    }

    console.log(expense);
}


exports.getExpense = async (req, res) => {
    try {
        const expenses = await ExpenseSchema.find().sort({ date: 1 });
        res.status(200).json(expenses)
    } catch (error) {
        res.status(500).json({message: "Server Error!"})
    }
}

exports.deleteExpense = async (req, res) => {
    const {id} = req.params;
    console.log(req.params)
    ExpenseSchema.findByIdAndDelete(id)
        .then((expense) => {
            res.status(200).json({message: "Expense deleted!"})
        })
        .catch((error) => {
            res.status(500).json({message: "Server Error!"})
        })
}