const { addExpense, getExpense, deleteExpense } = require("../controllers/expense")
const {addIncome, getIncomes, deleteIncome} = require("../controllers/income")
const { updateName, getName, checkIfName } = require("../controllers/user")
const router = require("express").Router()



router.post("/add-income", addIncome)
    .get("/get-incomes", getIncomes)
    .delete("/delete-income/:id", deleteIncome)
    .post("/add-expense", addExpense)
    .get("/get-expenses", getExpense)
    .delete("/delete-expense/:id", deleteExpense)
    .post("/change-name", updateName)
    .get("/get-name", getName)
    .get("/check-name", checkIfName)

module.exports = router