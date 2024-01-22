const UserSchema = require("../models/userModel")


exports.updateName = async (req, res) => {
    const {name} = req.body
    //console.log(req.body);
    //console.log("________________________________");
    try {
        const existingData = await UserSchema.findOne();
        if(name === "") {
            return res.status(400).json({message: "Cannot leave name as empty!"})
        }
        existingData.name = name;
        await existingData.save();
        //console.log(existingData);
        res.status(200).json({message: "Success!"})
    } catch (error) {
        //console.log(error);
        res.status(500).json({message: error})
    }
}

exports.checkIfName = async (req, res) => {
    try {
        const existingData = await UserSchema.findOne();
        if (!existingData) {
            const dummyData = new UserSchema({ name: 'Enter your name' });
            await dummyData.save();
            res.status(200).json({message: "Success!"})
        }
    } catch (error) {
        res.status(500).json({message: error})
    }
}

exports.getName = async (req, res) => {
    const {name} = req.body

    const user = UserSchema({
        name
    });

    try {
        const users = await UserSchema.findOne();
        res.status(200).json(users)
    } catch (error) {
        res.status(500).json({message: "Server Error!"})
    }
}

