import React, {useState} from "react"

export const Login = (props: any) => {
    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');

    const handleSubmit = (e: any) => {
        e.preventDefault();
        console.log(email);
    }

    return (
        <div className={"auth-form-container"}>
            <form className={"login-form"} onSubmit={handleSubmit}>
                <label htmlFor={"email"}>Email</label>
                <input value={email} onChange={(e) => setEmail(e.target.value)}
                       type={"email"} placeholder={"your-email@gmail.com"} id={"email"} name={"email"}/>

                <label htmlFor={"password"}>Password</label>
                <input value={password} onChange={(e) => setPassword(e.target.value)}
                       type={"password"} placeholder={"your-password!@#"} id={"password"} name={"password"}/>

                <button type={"submit"}>Log In</button>
            </form>
            <button className={"link-btn"} onClick={() => props.onFormSwitch('register')}>
                Don't have an account? Register here:
            </button>
        </div>
    )
}