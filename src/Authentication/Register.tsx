import React, {useState} from "react";

export const Register = (props: any) => {
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = async (e: any) => {
        e.preventDefault();
        try {
            const response = await fetch("http://127.0.0.1:5000/api/register", {
            method: "POST",
            headers: {
            "Content-Type": "application/json",
            },
            body: JSON.stringify({ firstName, lastName, email, password }),
        });
            const data = await response.json();
            console.log(data);
        } catch (error) {
            console.error("Error:", error);
        }
        props.onFormSwitch('state');
    }

    const handleBack = (e: any) => {
        e.preventDefault();
        props.onFormSwitch('home');
    }

    return (
        <div className={"auth-form-container"}>
            <form className={"register-form"} onSubmit={handleSubmit}>
                <label htmlFor={"First Name"}>First Name</label>
                <input value={firstName} onChange={(e) => setFirstName(e.target.value)}
                       type={"text"} placeholder={"First Name"} id={"firstName"} name={"firstName"}/>

                <label htmlFor={"Last Name"}>Last Name</label>
                <input value={lastName} onChange={(e) => setLastName(e.target.value)}
                       type={"text"} placeholder={"Last Name"} id={"lastName"} name={"lastName"}/>

                <label htmlFor={"email"}>Email</label>
                <input value={email} onChange={(e) => setEmail(e.target.value)}
                       type={"email"} placeholder={"your-email@gmail.com"} id={"email"} name={"email"}/>

                <label htmlFor={"password"}>Password</label>
                <input value={password} onChange={(e) => setPassword(e.target.value)}
                       type={"password"} placeholder={"your-password!@#"} id={"password"} name={"password"}/>

                <button type={"submit"}>Create Account</button>
            </form>
            <button type={"reset"} className={"link-btn"} onClick={() => props.onFormSwitch('login')}>
                Already have an account? Login here:
            </button>
            <button onClick={handleBack}>Back</button>
        </div>
    )
}