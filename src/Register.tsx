import React, {useState} from "react";

export const Register = (props: any) => {
    const [firstName, setFirstName] = useState('');
    const [lastName, setLastName] = useState('');
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = (e: any) => {
        e.preventDefault();
        console.log(firstName);
        console.log(lastName);
        console.log(email);
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

                <button type={"submit"} onClick={() => props}>Create Account</button>
            </form>
            <button className={"link-btn"} onClick={() => props.onFormSwitch('login')}>Already have an account? Login here:</button>
        </div>
    )
}